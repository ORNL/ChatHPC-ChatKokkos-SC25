#!/usr/bin/env bash
GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -e

echo '*** Running run_all.sh ***'

echo
echo "Running 1_setup.sh"
$GIT_ROOT/1_setup.sh

echo
echo "Running 2_train.sh"
$GIT_ROOT/2_train.sh

echo
echo "Running 3_verify.sh"
$GIT_ROOT/3_verify.sh

echo
echo "Running 4_evaluate.sh"
$GIT_ROOT/4_evaluate.sh

echo
echo "Running 5_evaluate_baseline.sh"
$GIT_ROOT/5_evaluate_baseline.sh