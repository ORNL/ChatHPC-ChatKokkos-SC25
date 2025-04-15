#!/usr/bin/env bash
GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo '*** Running 5_evaluate_baseline.sh ***'
echo
echo '*** Ensure commands are running in correct directory. ***'
echo cd $SCRIPT_DIR
cd $SCRIPT_DIR

echo
echo '*** Evaluate code-llama base model ***'
echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_initial.json base test --save_results_file evaluation/code_llama_base_results.json C2_Kokkos_Dataset/kokkos_testing.yaml
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_initial.json base test --save_results_file evaluation/code_llama_base_results.json C2_Kokkos_Dataset/kokkos_testing.yaml
echo "uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc-data-to-md evaluation/code_llama_base_results.json > evaluation/code_llama_base_results.md"
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc-data-to-md evaluation/code_llama_base_results.json > evaluation/code_llama_base_results.md

echo
echo '*** Evaluate OpenAI gpt-4o base model ***'
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set."
    echo "Please export your key if you would like to evauate the OpenAI gpt-4o base model."
    echo "export OPENAI_API_KEY=your-key-here"
    exit 1
else
    echo "OPENAI_API_KEY is set."
    echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_initial.json openai test --model gpt-4o --save_results_file evaluation/openai_gpt-4o_base_results.json C2_Kokkos_Dataset/kokkos_testing.yaml
    uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc --config config_initial.json openai test --model gpt-4o --save_results_file evaluation/openai_gpt-4o_base_results.json C2_Kokkos_Dataset/kokkos_testing.yaml
    echo "uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc-data-to-md evaluation/openai_gpt-4o_base_results.json > evaluation/openai_gpt-4o_base_results.md"
    uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc-data-to-md evaluation/openai_gpt-4o_base_results.json > evaluation/openai_gpt-4o_base_results.md
fi

