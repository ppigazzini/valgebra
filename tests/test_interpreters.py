"""The module loads in the main interpreter of a process, once.

The frontend reads `typing`, `annotated_types`, `dataclasses` and the module's
own markers into cells the process keeps. A subinterpreter holds its own copies
of those modules, and an interpreter started again after `Py_FinalizeEx` would
read the cells through objects the first one freed, so the import refuses both,
and goes on working in the interpreter that loaded it. Each check runs in a
child process, which a subinterpreter or an embedder needs of its own.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Annotated, Any

import pytest

from valgebra import Regex, Validator

pytestmark = pytest.mark.skipif(
    sys.implementation.name != "cpython",
    reason="only CPython runs a second interpreter in one process",
)

# A legacy subinterpreter, the kind `mod_wsgi` runs an application group in:
# it shares the process's extension modules, so the import reaches the module.
_SUBINTERPRETER = """
import sys
try:
    import _interpreters as interpreters
    created = interpreters.create("legacy")
except ImportError:
    import _xxsubinterpreters as interpreters
    created = interpreters.create(isolated=False)
CODE = '''
try:
    import valgebra
except ImportError as error:
    print("refused:", error)
else:
    print("loaded")
'''
FIRST = sys.argv[1]
if FIRST == "main":
    from typing import Annotated, Any
    from valgebra import Regex, Validator
interpreters.run_string(created, CODE)
from typing import Annotated, Any
from valgebra import Regex, Validator
print(
    "main:",
    Validator(Any).is_valid(1),
    Validator(Annotated[str, Regex("a")]).is_valid("b"),
)
interpreters.destroy(created)
print("after:", Validator(list[int]).is_valid(["x"]))
"""


@pytest.mark.parametrize("first", ["main", "subinterpreter"])
def test_a_subinterpreter_is_refused_and_the_main_one_answers(first: str) -> None:
    """A subinterpreter's import raises, whichever imports first.

    The main interpreter loads the module and answers as this one does,
    before the subinterpreter asks, after it, and after it is destroyed.
    """
    done = subprocess.run(  # noqa: S603 -- fixed interpreter, in-repo program
        [sys.executable, "-c", _SUBINTERPRETER, first],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0].startswith("refused: valgebra loads in the main interpreter"), lines
    alone = (
        Validator(Any).is_valid(1),
        Validator(Annotated[str, Regex("a")]).is_valid("b"),
        Validator(list[int]).is_valid(["x"]),
    )
    assert lines[1:] == [f"main: {alone[0]} {alone[1]}", f"after: {alone[2]}"], lines


_AGAIN = """
import importlib
import sys
import valgebra
for name in [name for name in sys.modules if name.startswith("valgebra")]:
    del sys.modules[name]
import valgebra
importlib.reload(valgebra)
print(valgebra.Validator(int).is_valid(1))
"""


def test_a_second_import_in_the_interpreter_that_loaded_it_works() -> None:
    """A cleared `sys.modules` and a reload import the module again."""
    done = subprocess.run(  # noqa: S603 -- fixed interpreter, in-repo program
        [sys.executable, "-c", _AGAIN],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert (done.returncode, done.stdout.strip()) == (0, "True"), done.stderr


# An interpreter with no `_thread._is_main_interpreter` asks
# `_xxsubinterpreters`, as 3.11 does; one where that module cannot be imported
# has one interpreter, and an import of it failing some other way raises. The
# refusing finder goes first, ahead of the finders that hold the module through
# 3.12; the child runs one thread, so no other import is iterating the list.
_NO_PROBE = """
import _thread, sys
vars(_thread).pop("_is_main_interpreter", None)
class Refusing:
    def find_spec(self, name, path=None, target=None):
        if name == "_xxsubinterpreters":
            raise RuntimeError("the probe module raises")
        return None
if sys.argv[1] == "missing":
    sys.modules["_xxsubinterpreters"] = None
else:
    sys.modules.pop("_xxsubinterpreters", None)
    sys.meta_path.insert(0, Refusing())
try:
    import valgebra
except RuntimeError as raised:
    print("raised:", raised)
else:
    print("loaded:", valgebra.Validator(int).is_valid(1))
"""


@pytest.mark.parametrize(
    ("probe", "read"),
    [("missing", "loaded: True"), ("raising", "raised: the probe module raises")],
)
def test_an_interpreter_with_no_subinterpreter_module_is_the_main_one(
    probe: str, read: str
) -> None:
    """With no way to ask, the interpreter is the only one, as on PyPy.

    A module that cannot be imported says so; one whose import fails any other
    way is an error of the import, which is raised rather than read as an
    answer.
    """
    done = subprocess.run(  # noqa: S603 -- fixed interpreter, in-repo program
        [sys.executable, "-c", _NO_PROBE, probe],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert (done.returncode, done.stdout.strip()) == (0, read), done.stderr


# Three lives of one embedded interpreter, each importing the module and
# building a validator; a `0` from the run is the import refused, which is how
# the second and third lives end.
_EMBEDDER = r"""
#include <Python.h>
#include <stdio.h>
int main(void) {
    for (int life = 0; life < 3; life++) {
        Py_Initialize();
        int failed = PyRun_SimpleString(
            "from typing import Annotated\n"
            "from valgebra import Regex, Validator\n"
            "print(Validator(Annotated[str, Regex('a')]).is_valid('b'), flush=True)\n");
        printf("life %d: %s\n", life, failed ? "refused" : "ran");
        fflush(stdout);
        if (Py_FinalizeEx() < 0) {
            return 2;
        }
    }
    return 0;
}
"""


def _embedding() -> tuple[str, str] | None:
    """Return a C compiler and the shared library to embed, or `None`."""
    compiler = shutil.which("cc")
    libdir = sysconfig.get_config_var("LIBDIR")
    library = sysconfig.get_config_var("LDLIBRARY")
    if (
        sys.platform != "linux"
        or compiler is None
        or not sysconfig.get_config_var("Py_ENABLE_SHARED")
        or not libdir
        or not library
        or not (Path(libdir) / library).exists()
    ):
        return None
    return compiler, libdir


@pytest.mark.skipif(
    _embedding() is None, reason="needs a C compiler and a shared libpython"
)
def test_an_interpreter_started_again_refuses_the_import(tmp_path: Path) -> None:
    """`Py_Initialize` after `Py_FinalizeEx` is refused the import, and ends cleanly.

    The first life answers as one interpreter does. A second life read the
    process's cells through objects the first one freed, and ended the process.
    """
    found = _embedding()
    assert found is not None
    compiler, libdir = found
    source = tmp_path / "embed.c"
    source.write_text(_EMBEDDER)
    program = tmp_path / "embed"
    version = sysconfig.get_config_var("LDVERSION") or sysconfig.get_python_version()
    subprocess.run(  # noqa: S603 -- the compiler the test found, its own source
        [
            compiler,
            str(source),
            "-o",
            str(program),
            "-I",
            sysconfig.get_path("include"),
            "-L",
            libdir,
            f"-lpython{version}",
            f"-Wl,-rpath,{libdir}",
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    environment = {
        **os.environ,
        "PYTHONHOME": sys.base_prefix,
        "PYTHONPATH": os.pathsep.join(path for path in sys.path if path),
    }
    done = subprocess.run(  # noqa: S603 -- the program this test built
        [str(program)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=environment,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        str(Validator(Annotated[str, Regex("a")]).is_valid("b")),
        "life 0: ran",
        "life 1: refused",
        "life 2: refused",
    ], done.stdout
