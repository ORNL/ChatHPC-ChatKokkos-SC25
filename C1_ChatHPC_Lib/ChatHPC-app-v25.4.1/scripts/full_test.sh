#!/usr/bin/env bash
GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pushd $GIT_ROOT

echo chathpc --config tests/files/config.json train
chathpc --config tests/files/config.json train
echo
echo chathpc --config tests/files/config.json verify
chathpc --config tests/files/config.json verify
echo
echo ./scripts/0_test_training.sh
./scripts/0_test_training.sh
echo
echo ./scripts/1_verify_training.sh
./scripts/1_verify_training.sh
echo
echo ./scripts/2_copy_jupyter_output.sh
./scripts/2_copy_jupyter_output.sh

popd
