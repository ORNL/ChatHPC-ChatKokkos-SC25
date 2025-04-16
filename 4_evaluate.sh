#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo '*** Running 4_evaluate.sh ***'
echo
echo '*** Ensure commands are running in correct directory. ***'
echo cd $SCRIPT_DIR
cd $SCRIPT_DIR

echo
echo '*** Evaluating Chatkokkos Initial ***'
echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_initial.json test --save_results_file evaluation/ChatKokkos_initial_results.json C2_Kokkos_Dataset/kokkos_testing.yaml
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_initial.json test --save_results_file evaluation/ChatKokkos_initial_results.json C2_Kokkos_Dataset/kokkos_testing.yaml
echo "uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc-data-to-md evaluation/ChatKokkos_initial_results.json > evaluation/ChatKokkos_initial_results.md"
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc-data-to-md evaluation/ChatKokkos_initial_results.json > evaluation/ChatKokkos_initial_results.md

echo
echo '*** Evaluating Chatkokkos Refinement ***'
echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_refinement.json test --save_results_file evaluation/ChatKokkos_refinement_results.json C2_Kokkos_Dataset/kokkos_testing.yaml
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_refinement.json test --save_results_file evaluation/ChatKokkos_refinement_results.json C2_Kokkos_Dataset/kokkos_testing.yaml
echo "uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc-data-to-md evaluation/ChatKokkos_refinement_results.json > evaluation/ChatKokkos_refinement_results.md"
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc-data-to-md evaluation/ChatKokkos_refinement_results.json > evaluation/ChatKokkos_refinement_results.md