# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

"""Backend selection semantics.

Selection happens once, at import time, so each case runs in its own
subprocess with its own environment rather than trying to reload the package
in-process.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import a5
from a5._backend import native_module

HAVE_NATIVE = native_module() is not None

# The directory holding the `a5` package this test session actually imported.
# Passing exactly this to the subprocesses makes them resolve `a5` the same way,
# whether that is a source checkout or an installed wheel.
A5_PARENT_DIR = str(Path(a5.__file__).resolve().parent.parent)

# Setting a module to None in sys.modules makes importing it raise ImportError,
# which is how the "no compiled extension" cases are simulated without
# uninstalling anything.
BLOCK_NATIVE = 'import sys; sys.modules["a5._a5"] = None; '


def run(code, **env_overrides):
    """Run `code` in a subprocess with a controlled A5 environment.

    Runs from a neutral directory with PYTHONPATH pointing at the parent of the
    `a5` package this session imported, so the subprocess resolves `a5` exactly
    the way the test session did -- the source tree for an editable install,
    site-packages when the suite runs against a built wheel. Inheriting the full
    sys.path instead would let a source checkout shadow an installed wheel and
    hide its extension module.
    """
    env = dict(os.environ)
    for key in ('A5_BACKEND', 'A5_EXPECT_BACKEND'):
        env.pop(key, None)
    env['PYTHONPATH'] = A5_PARENT_DIR
    env.update({k: v for k, v in env_overrides.items() if v is not None})
    return subprocess.run(
        [sys.executable, '-c', code],
        cwd=tempfile.gettempdir(),
        env=env,
        capture_output=True,
        text=True,
    )


def backend_under(**env_overrides):
    result = run('import a5; print(a5.get_backend())', **env_overrides)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


class TestSelection:
    def test_default_is_pure_python_in_0x(self):
        # 0.x keeps the pre-existing behaviour as the default so that upgrading
        # cannot silently change which implementation runs. This assertion is
        # the tripwire for the 1.0 flip: when `_DEFAULT_BACKEND` becomes
        # 'auto', this test has to be updated deliberately.
        assert backend_under() == 'python'

    def test_explicit_python(self):
        assert backend_under(A5_BACKEND='python') == 'python'

    @pytest.mark.skipif(not HAVE_NATIVE, reason='compiled extension not built')
    def test_explicit_rust(self):
        assert backend_under(A5_BACKEND='rust') == 'rust'

    @pytest.mark.skipif(not HAVE_NATIVE, reason='compiled extension not built')
    def test_auto_prefers_rust_when_available(self):
        assert backend_under(A5_BACKEND='auto') == 'rust'

    def test_backend_is_case_and_space_insensitive(self):
        assert backend_under(A5_BACKEND=' PYTHON ') == 'python'

    def test_empty_backend_falls_through_to_the_default(self):
        assert backend_under(A5_BACKEND='') == 'python'

    def test_invalid_backend_is_rejected(self):
        result = run('import a5', A5_BACKEND='fortran')
        assert result.returncode != 0
        assert 'A5_BACKEND must be one of' in result.stderr


class TestFallback:
    def test_auto_falls_back_when_extension_missing(self):
        result = run(BLOCK_NATIVE + 'import a5; print(a5.get_backend())', A5_BACKEND='auto')
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'python'

    def test_default_falls_back_when_extension_missing(self):
        result = run(BLOCK_NATIVE + 'import a5; print(a5.get_backend())')
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'python'

    def test_explicit_rust_raises_when_extension_missing(self):
        # An opt-in must never degrade silently -- otherwise someone testing the
        # compiled path would benchmark and validate pure Python by accident.
        result = run(BLOCK_NATIVE + 'import a5', A5_BACKEND='rust')
        assert result.returncode != 0
        assert 'A5_BACKEND=rust was requested' in result.stderr
        assert 'ImportError' in result.stderr

    def test_fallback_still_serves_the_whole_api(self):
        code = BLOCK_NATIVE + (
            'import a5;'
            'assert a5.get_backend() == "python";'
            'assert all(hasattr(a5, name) for name in a5.__all__);'
            'print(a5.lonlat_to_cell((-1.0, 51.0), 10))'
        )
        result = run(code)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(a5.lonlat_to_cell((-1.0, 51.0), 10))


class TestSurface:
    """The two import branches in a5/__init__.py must not drift apart."""

    def test_public_names_are_identical_across_backends(self):
        code = 'import a5; print(",".join(sorted(a5.__all__)))'
        python_names = run(code, A5_BACKEND='python').stdout.strip()
        assert python_names
        if HAVE_NATIVE:
            rust_names = run(code, A5_BACKEND='rust').stdout.strip()
            assert rust_names == python_names

    def test_every_public_name_is_bound(self):
        for name in a5.__all__:
            assert getattr(a5, name, None) is not None, name

    @pytest.mark.skipif(not HAVE_NATIVE, reason='compiled extension not built')
    def test_native_module_is_reachable_regardless_of_selection(self):
        # The differential tests and benchmarks need both implementations even
        # when the selected backend is pure Python.
        code = (
            'from a5._backend import native_module, get_backend;'
            'print(get_backend(), native_module() is not None)'
        )
        result = run(code, A5_BACKEND='python')
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'python True'


def test_get_backend_matches_module_constant():
    from a5._backend import BACKEND

    assert a5.get_backend() == BACKEND
    assert BACKEND in ('rust', 'python')
