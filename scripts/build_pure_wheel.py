"""Build the pure-Python wheel (py3-none-any) that ships next to the platform wheels.

pip prefers a platform wheel wherever one matches, so this wheel is only
installed where none does (PyPy, platforms outside the wheels.yml matrix).
There `a5` falls back to the pure-Python backend, rather than pip building the
sdist, which needs a Rust toolchain.

maturin always compiles the extension, so the wheel is built with hatchling
instead, from a copy of the project whose pyproject.toml has its build backend
swapped. Everything else ([project] metadata included) is taken verbatim.

    python scripts/build_pure_wheel.py [out_dir]   # default: dist/
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BUILD_SYSTEM = '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n'

WHEEL_TARGET = '\n[tool.hatch.build.targets.wheel]\npackages = ["a5"]\n'

# Local builds leave the compiled extension and bytecode in the source tree
IGNORE = shutil.ignore_patterns('__pycache__', '*.py[oc]', '*.so', '*.pyd')


def main(out_dir):
    pyproject = (ROOT / 'pyproject.toml').read_text()
    # The [build-system] table runs until the next line opening a table
    pyproject, count = re.subn(r'^\[build-system\]\n(?:(?!\[).*\n)*', BUILD_SYSTEM, pyproject, flags=re.M)
    if count != 1:
        sys.exit('pyproject.toml: expected exactly one [build-system] table, found {}'.format(count))

    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp)
        shutil.copytree(ROOT / 'a5', project / 'a5', ignore=IGNORE)
        for name in ('README.md', 'LICENSE'):
            shutil.copy(ROOT / name, project / name)
        (project / 'pyproject.toml').write_text(pyproject + WHEEL_TARGET)
        subprocess.run(
            [sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--wheel-dir', str(out_dir.resolve()), str(project)],
            check=True,
        )


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'dist')
