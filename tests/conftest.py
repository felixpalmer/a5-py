# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

"""Backend reporting and guarding for the fixture suite.

The suite is run once per backend (see .github/workflows/test.yml). Everything
imported from the `a5` namespace therefore exercises whichever implementation is
selected, which is what makes the pure-Python port a continuous differential
test of the compiled one rather than a parallel mirror we hope stays in sync.

Two safeguards:

* The active backend is printed in the pytest header, so a log always says which
  implementation produced the result.
* If ``A5_EXPECT_BACKEND`` is set and does not match, collection fails. Without
  it a missing or broken extension module would silently fall back to pure
  Python and the "rust" CI job would quietly test Python twice.
"""

import os
import sys
from pathlib import Path

# Several test modules do `from tests.matchers import ...`, which needs this
# file's parent directory importable. That happens for free when pytest is run
# from the repository root, but not when the suite is run against an installed
# wheel from elsewhere (see the cibuildwheel test-command). Appended rather than
# prepended: `a5` must keep resolving to whatever the caller installed, not to
# the source tree sitting next to `tests/`.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import a5  # noqa: E402


def pytest_report_header(config):
    return 'a5 backend: {}'.format(a5.get_backend())


def pytest_collection(session):
    expected = os.environ.get('A5_EXPECT_BACKEND', '').strip().lower()
    actual = a5.get_backend()
    if expected and expected != actual:
        raise pytest.UsageError(
            'A5_EXPECT_BACKEND={!r} but the active backend is {!r}. The compiled '
            'extension is probably missing; build it with '
            '`maturin develop --release`.'.format(expected, actual)
        )
