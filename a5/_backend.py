# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

"""Backend selection for the A5 public API.

A5 ships two interchangeable implementations of the same public API:

``python``
    The pure-Python implementation under :mod:`a5.core`, :mod:`a5.traversal`
    and :mod:`a5.regions`. It has no build step and no dependencies, and it is
    the reference implementation: every cross-port fixture is written against
    it, and :mod:`tests.test_differential` checks the compiled backend against
    it cell-for-cell.

``rust``
    PyO3 bindings to the a5-rs crate, exposed as the private extension module
    ``a5._a5`` and adapted back to the Python signatures by :mod:`a5._native`.
    Roughly two orders of magnitude faster; present only in the platform wheels.

Selection is controlled by ``A5_BACKEND``:

``auto``
    Use ``rust`` if the extension module imports, otherwise ``python``.
``rust``
    Require the extension module; raise :class:`ImportError` if it is missing,
    so an opt-in never degrades to pure Python without saying so.
``python``
    Always use the pure-Python implementation.

``A5_PURE_PYTHON=1`` is accepted as an alias for ``A5_BACKEND=python``.
``A5_BACKEND`` wins if both are set.

The backend is resolved once, at import time. Changing the environment
afterwards has no effect.
"""

import os
from typing import Optional

__all__ = ['get_backend', 'native_module']

# The default when neither environment variable is set.
#
# 0.x ships pure Python as the default so that upgrading cannot change results
# or performance characteristics under anyone's feet, with ``A5_BACKEND=rust``
# as the opt-in for testing the compiled path. This flips to ``'auto'`` in 1.0,
# at which point the compiled backend becomes the default wherever a wheel
# provides it. That release note is the only thing this constant gates.
_DEFAULT_BACKEND = 'python'

_VALID_BACKENDS = ('auto', 'rust', 'python')

_TRUTHY = ('1', 'true', 'yes', 'on')

_native = None  # type: Optional[object]
_native_error = None  # type: Optional[BaseException]


def _load_native():
    """Import the compiled extension module, caching success and failure."""
    global _native, _native_error
    if _native is None and _native_error is None:
        try:
            from . import _a5  # type: ignore[attr-defined]
        except ImportError as exc:  # pragma: no cover - depends on the build
            _native_error = exc
        else:
            _native = _a5
    return _native


def native_module():
    """Return the compiled extension module, or ``None`` if it is not built.

    Ignores ``A5_BACKEND`` entirely -- this is how the differential tests and
    the benchmark suite reach the compiled implementation even when the
    selected backend is pure Python.
    """
    return _load_native()


def _requested():
    """Resolve the requested backend name from the environment."""
    requested = os.environ.get('A5_BACKEND', '').strip().lower()
    if requested:
        if requested not in _VALID_BACKENDS:
            raise ValueError(
                'A5_BACKEND must be one of {}, got {!r}'.format(
                    ', '.join(_VALID_BACKENDS), requested
                )
            )
        return requested
    if os.environ.get('A5_PURE_PYTHON', '').strip().lower() in _TRUTHY:
        return 'python'
    return _DEFAULT_BACKEND


def _resolve():
    """Pick the backend to use, importing the extension module if needed."""
    requested = _requested()
    if requested == 'python':
        return 'python'
    if _load_native() is not None:
        return 'rust'
    if requested == 'rust':
        raise ImportError(
            'A5_BACKEND=rust was requested but the compiled extension module '
            'a5._a5 could not be imported ({}). Install a platform wheel, or '
            'build it in place with `maturin develop --release`.'.format(_native_error)
        )
    return 'python'


BACKEND = _resolve()


def get_backend():
    # type: () -> str
    """Return the active backend, either ``'rust'`` or ``'python'``.

    This is specific to the Python port -- the TypeScript and Rust
    implementations have a single backend each.
    """
    return BACKEND
