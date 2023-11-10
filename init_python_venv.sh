#!/bin/bash

if ! git rev-parse --git-dir > /dev/null 2>&1; then
  : # This is not a valid git repository and will fail due to scm erroring, so tell the user
    echo "You have not initialised a git repo. Please run ./init_git.sh first and try again"
    exit
fi

_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pushd ${_SCRIPT_DIR} # cd to script directory

# https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euxo pipefail

echo "Setting up virtualenv"

python -m venv .venv
.venv/bin/pip install --upgrade pip  # >= 21.3.1
.venv/bin/pip install -e .[dev] --constraint lockfiles/3.11/lockfile.txt


popd # return to original directory
unset _SCRIPT_DIR
