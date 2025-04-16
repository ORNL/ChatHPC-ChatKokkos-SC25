#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo '*** Running 2_train.sh ***'
echo
echo '*** Ensure commands are running in correct directory. ***'
echo cd $SCRIPT_DIR
cd $SCRIPT_DIR

echo
echo '*** Training Chatkokkos Initial ***'
echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_initial.json train
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_initial.json train

echo
echo '*** Training Chatkokkos Refinement ***'
echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_refinement.json train
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_refinement.json train