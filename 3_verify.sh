#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo '*** Running 3_verify.sh ***'
echo
echo '*** Ensure commands are running in correct directory. ***'
echo cd $SCRIPT_DIR
cd $SCRIPT_DIR

echo
echo '*** Verifying Chatkokkos Initial ***'
echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_initial.json verify
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_initial.json verify

echo
echo '*** Verifying Chatkokkos Refinement ***'
echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_refinement.json verify
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_refinement.json verify