#!/usr/bin/env bash
GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -x

source $GIT_ROOT/.venv/bin/activate

pip install papermill

pushd $GIT_ROOT/examples

python $GIT_ROOT/scripts/utils/verify_app.py > verify_training.txt

popd
