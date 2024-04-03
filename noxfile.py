import pathlib
import tempfile
from typing import Optional

import nox

SUPPORTED_PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]

nox.options.envdir = ".cache/nox"
nox.options.reuse_existing_virtualenvs = False
nox.options.default_venv_backend = "virtualenv"
nox.options.sessions = ["formatting_check-3.8", "typing_check-3.8", "tests_run_latest-3.8"]

SOURCE_FILES = ("src/", "tests/", "noxfile.py")

# Artefacts folders
PINNED_VERSIONS = "pinned-versions"
COVERAGE_DIR = ".coverage"

TEST_REPORTS_DIR = "test-reports"

# all external utility tools used by our nox sessions
BUILD_TOOLS = ["build"]
COVERAGE_TOOLS = ["coverage[toml]", "coverage-badge"]
FORMATTING_TOOLS = ["black[jupyter]~=23.0"]
LINTING_TOOLS = ["ruff~=0.0.292"]
LOCKFILE_TOOLS = ["pip-tools>=7.0.0"]  # default --resolver=backtracking

EXTRAS = [None, "openmm", "rdkit", "ase", "openeye"]
DONT_TEST = [None, "openeye", "openmm"]

def resolve_lockfile_path(python_version: str, extra: Optional[str] = None, rootdir: str = PINNED_VERSIONS) -> pathlib.Path:
    """Resolves the expected lockfile path for a given python version and extra."""
    lockfile_name = f"lockfile.{extra or 'core'}.txt"
    return pathlib.Path(rootdir) / python_version / lockfile_name


def resolve_coverage_datafile_path(python_version: str, extra: Optional[str] = None) -> pathlib.Path:
    """Resolves the expected coverage data_file path for a given python version and extra."""
    coverage_datafile_name = f".coverage.{extra or 'core'}"
    return pathlib.Path(COVERAGE_DIR) / python_version / coverage_datafile_name


def resolve_junitxml_path(python_version: str, extra: Optional[str] = None) -> pathlib.Path:
    """Resolves the output pytest junitxml reports path for a given python version and extra."""
    junitxml_report_name = f".junitxml.{extra or 'core'}.xml"
    return pathlib.Path(TEST_REPORTS_DIR) / python_version / junitxml_report_name


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def coverage_report(session: nox.Session) -> None:
    """Generate coverage reports.

    This session is usually triggered following a pytest coverage session generating
    the coverage data files to build the reports on. The directory containing
    the coverage data files should be provided as posarg.

    Examples:
        nox -s coverage_report-3.11 -- .coverage/3.11
    """

    session.install(*COVERAGE_TOOLS)

    # Combine coverage output data_files
    datafiles_dir = session.posargs[0]
    merged_data_file = f"{datafiles_dir}/.coverage"
    session.run("coverage", "combine", "--data-file", merged_data_file, "--keep", datafiles_dir)

    # Output coverage reports in same folder (for convenience)
    coverage_reports_dir = datafiles_dir
    session.run("coverage", "report", "--data-file", merged_data_file)
    session.run("coverage", "html", "--data-file", merged_data_file, "-d", f"{coverage_reports_dir}/html")
    session.run("coverage", "xml", "--data-file", merged_data_file, "-o", f"{coverage_reports_dir}/coverage.xml")
    session.run("coverage", "json", "--data-file", merged_data_file, "-o", f"{coverage_reports_dir}/coverage.json")

    session.notify(f"coverage_build_badge-{session.python}", [coverage_reports_dir])


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def coverage_build_badge(session: nox.Session) -> None:
    """Generate a coverage badge.

    This session is usually triggered following a pytest coverage session generating
    the coverage data files for which to build the badge. The directory containing
    the final .coverage data file should be provided as posarg.

    Examples:
        nox -s coverage_build_badge-3.11 -- .coverage/3.11
    """

    # coverage-badge only works from the same directory where the .coverage
    # data file is located.
    data_file_dir = session.posargs[0]
    session.chdir(data_file_dir)

    badge_filename = "coverage.svg"

    # cleanup old badge
    session.run("rm", "-rf", badge_filename, external=True)

    session.install(*COVERAGE_TOOLS)
    session.run("coverage-badge", "-o", badge_filename)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def dist_build(session: nox.Session) -> None:
    """Build distributions (sdist and wheel).

    The distribution packages are built using PyPA's `build`_ build frontend.
    This is the recommended way of building python packages, avoiding direct
    calls to the build backend. Legacy calls like ``$ python setup.py build``
    are now deprecated.

    .. _build:
            https://pypa-build.readthedocs.io/en/latest/

    Examples:
        nox -s dist_build-3.11
    """

    session.run("rm", "-rf", "dist", external=True)
    session.install(*BUILD_TOOLS)
    session.run("python", "-m", "build")


@nox.session(venv_backend="conda", python=SUPPORTED_PYTHON_VERSIONS)
def docs_build(session: nox.Session) -> None:
    """Build sphinx documentation and API docs.

    Examples:
        nox -s docs_build-3.11
    """

    lockfile_path = resolve_lockfile_path(python_version=session.python)
    session.install(".[docs]", "--constraint", lockfile_path)

    # # Build API docs
    # apidoc_cmd = "sphinx-apidoc -f -o docs/source/pages/reference/api src/physicsml --implicit-namespaces"
    # session.run(*apidoc_cmd.split(" "))

    # wipe artefacts from previous runs, in case there are any
    session.run("rm", "-rf", "docs/build/html", external=True)

    # -a -E flags make sure things are built from scratch
    build_cmd = "sphinx-build -a -E docs/source/ docs/build/html"

    # # Run doctests in the documentation
    # session.run(*build_cmd.split(" "), "-b", "doctest")

    # Build HTML pages
    session.run(*build_cmd.split(" "), "-b", "html")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def formatting_check(session: nox.Session) -> None:
    """Check codebase formatting.

    Examples:
        nox -s formatting_check-3.11
    """
    session.install(*FORMATTING_TOOLS)
    session.run("black", "--check", "--diff", ".")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def formatting_fix(session: nox.Session) -> None:
    """Fix codebase formatting.

    Examples:
        nox -s formatting_fix-3.11
    """
    session.install(*FORMATTING_TOOLS)
    session.run("black", ".")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def linting_check(session: nox.Session) -> None:
    """Check codebase lint quality.

    Examples:
        nox -s linting_check-3.11
    """
    session.install(*LINTING_TOOLS)
    session.run("ruff", "check", *session.posargs, *SOURCE_FILES)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def linting_fix(session: nox.Session) -> None:
    """Fix codebase lint quality where possible.

    Examples:
        nox -s linting_fix-3.11
    """
    session.install(*LINTING_TOOLS)
    session.run("ruff", "--fix", *session.posargs, *SOURCE_FILES)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def typing_check(session: nox.Session) -> None:
    """Check codebase type annotations.

    Examples:
        nox -s typing_check-3.11
        nox -s typing_check-3.11 -- --python-version 3.11
    """

    lockfile_path = resolve_lockfile_path(python_version=session.python)
    session.install(".[tests,typing]",  "--constraint", lockfile_path)
    session.run("mypy", *session.posargs)


def generate_lockfile(session: nox.Session, extra: Optional[str], lockfile_path: pathlib.Path) -> None:
    """Generates a package dependencies' lockfile.

    Args:
        session: The nox.Session to use for the operation.
        extra: The name of an additional specific package extra to take into
            account when resolving dependencies. If not None, this will be used
            in addition to the usual 'docs' and 'tests' extras.
        lockfile_path: The path where to output the generated lockfile.
    """

    session.install(*LOCKFILE_TOOLS)

    package_extras = f"docs,tests,{extra}" if extra else "docs,tests"
    lockfile_path.parent.mkdir(parents=True, exist_ok=True)
    session.run(
        "pip-compile",
        "--verbose",
        f"--extra={package_extras}",
        "--strip-extras",
        "--no-emit-index-url",
        "--no-emit-trusted-host",
        "pyproject.toml",
        "-o",
        str(lockfile_path),
        "--upgrade",
        env={"CUSTOM_COMPILE_COMMAND": f"nox -s {session.name}"},
    )
    print(f"Lockfile generated at {str(lockfile_path)!r} ✨")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@nox.parametrize("extra", EXTRAS)
def dependencies_pin(session: nox.Session, extra: Optional[str]) -> None:
    """Generate pinned dependencies lockfiles.

    Examples:
        nox -s dependencies_pin-3.11
        nox -s "dependencies_pin-3.11(extra=None)"
    """
    output_lockfile_path = resolve_lockfile_path(python_version=session.python, extra=extra)
    generate_lockfile(session, extra=extra, lockfile_path=output_lockfile_path)


def run_tests(session: nox.Session, *args: str, extra: Optional[str], lockfile_path: pathlib.Path, notify: bool) -> None:
    """Runs tests.

    This includes running code snippets in our source code docstrings.

    Args:
        session: The nox.Session to use for the operation.
        extra: The name of the package extra for which to run tests. If None,
            only core tests will be collected.
        lockfile_path: The path to the lockfile to use for constraining the
            testing environment.
        notify: If True, coverage reporting sessions are queued up on success.
    """

    # Setup which files and tests to target
    # install test dependencies and extra dependencies
    if extra is not None:
        package_extras = ",".join(["tests", extra])

    else:
        package_extras = ",".join(["tests"])

    # all tests requiring the extra are in their own dir and
    tests_target_dirs = [f"tests/{extra}"]

    # Run tests
    session.install(f".[{package_extras}]", "--constraint", str(lockfile_path))

    coverage_datafile_path = resolve_coverage_datafile_path(python_version=session.python, extra=extra)
    junitxml_path = resolve_junitxml_path(python_version=session.python, extra=extra)
    session.run(
        "coverage",
        "run",
        f"--data-file={coverage_datafile_path}",
        "-m",
        "pytest",
        f"--junitxml={junitxml_path}",
        *args,
        *tests_target_dirs,
    )

    # for parametrised runs, this will get run once at the end of all of them
    if notify:
        datafiles_dir = str(coverage_datafile_path.parent)
        session.notify(f"coverage_report-{session.python}", [datafiles_dir])


@nox.session(venv_backend="conda", python=SUPPORTED_PYTHON_VERSIONS)
@nox.parametrize("extra", [e for e in EXTRAS if e not in DONT_TEST])
def tests_run_latest(session: nox.Session, extra: Optional[str]) -> None:
    """Run tests against latest available dependencies.

    Examples:
        nox -s tests_run_latest-3.11
        nox -s "tests_run_latest-3.11(extra=None)"
    """
    # Generate a scratch lock file with latest resolved dependencies
    #
    # we could have used session.create_tmp but that sets $TMPDIR which creates
    # problems with multiprocessing code: https://github.com/python/cpython/issues/93852
    with tempfile.TemporaryDirectory() as tmp:
        scratch_output_lockfile_path = resolve_lockfile_path(python_version=session.python, extra=extra, rootdir=tmp)
        generate_lockfile(session, extra=extra, lockfile_path=scratch_output_lockfile_path)

        run_tests(session, *session.posargs, extra=extra, lockfile_path=scratch_output_lockfile_path, notify=False)


@nox.session(venv_backend="conda", python=SUPPORTED_PYTHON_VERSIONS)
@nox.parametrize("extra", [e for e in EXTRAS if e not in DONT_TEST])
def tests_run_pinned(session: nox.Session, extra: Optional[str]) -> None:
    """Run tests against pinned dependencies.

    These should already be present. If not, they can be generated / updated
    by running the `dependencies_pin` session.

    Examples:
        nox -s tests_run_pinned-3.11
        nox -s "tests_run_pinned-3.11(extra=None)"
    """
    expected_lockfile_path = resolve_lockfile_path(python_version=session.python, extra=extra)
    run_tests(session, *session.posargs, extra=extra, lockfile_path=expected_lockfile_path, notify=True)
