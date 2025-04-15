#!/usr/bin/env bash
GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -x

source $GIT_ROOT/.venv/bin/activate

pip install papermill

pushd $GIT_ROOT/examples

for f in $GIT_ROOT/examples/*.ipynb
do
    echo papermill "$f" "${f%.*}_output.ipynb"
    papermill "$f" "${f%.*}_output.ipynb"
    echo jupyter nbconvert --to html "${f%.*}_output.ipynb"
    jupyter nbconvert --to html "${f%.*}_output.ipynb"
    $GIT_ROOT/scripts/utils/extract_responses.py "${f%.*}_output.ipynb" > "${f%.*}_output.txt"
done

CHATHPC_DATA_FILE="/home/7ry/Data/ellora/ChatKokkos-data/kokkos_dataset_before_reinforcement.json"\
    CHATHPC_FINETUNED_MODEL_PATH="./app/peft_adapter"\
    CHATHPC_MERGED_MODEL_PATH="./app/merged_adapters"\
    CHATHPC_TRAINING_OUTPUT_DIR="./app/kokkos-code-llama"\
    CHATHPC_PROMPT_TEMPLATE='You are a powerful LLM model for Kokkos called ChatKokkos created by ORNL. Your job is to answer questions about the Kokkos programming model. You are given a question and context regarding the Kokkos programming model.\n\nYou must output the answer the question.\n\n### Context:\n{{ context }}\n\n### Question:\n{{ question }}\n\n### Answer:\n{{ answer }}\n\n'\
    chathpc train

CHATHPC_DATA_FILE="/home/7ry/Data/ellora/ChatKokkos-data/kokkos_dataset_before_reinforcement.json"\
    CHATHPC_FINETUNED_MODEL_PATH="./app_old/peft_adapter"\
    CHATHPC_MERGED_MODEL_PATH="./app_old/merged_adapters"\
    CHATHPC_TRAINING_OUTPUT_DIR="./app_old/kokkos-code-llama"\
    CHATHPC_PROMPT_TEMPLATE='You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.\n\nYou must output the Kokkos question that answers the question.\n\n### Input:\n{{question}}\n\n### Context:\n{{context}}\n\n### Response:\n{{answer}}\n'\
    chathpc train

popd
