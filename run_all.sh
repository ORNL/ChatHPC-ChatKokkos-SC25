#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -e

echo '*** Running run_all.sh ***'

echo
echo "Running 1_setup.sh"
$SCRIPT_DIR/1_setup.sh

echo
echo "Running 2_train.sh"
$SCRIPT_DIR/2_train.sh

echo
echo "Running 3_verify.sh"
$SCRIPT_DIR/3_verify.sh

echo
echo "Running 4_evaluate.sh"
$SCRIPT_DIR/4_evaluate.sh

echo
echo "Running 5_evaluate_baseline.sh"
$SCRIPT_DIR/5_evaluate_baseline.sh