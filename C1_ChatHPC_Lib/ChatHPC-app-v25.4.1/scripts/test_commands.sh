#!/usr/bin/env bash
set -e

# Test help functions
echo '*** Test -h for scripts ***'

echo ChatHPC -h
ChatHPC -h
echo
echo chathpc -h
chathpc -h

# Test default arguments
echo
echo '*** Test basic script functionality ***'

echo CHATHPC_DATA_FILE="/home/7ry/Data/ellora/ChatKokkos-data/kokkos_dataset_before_reinforcement.json"\
    CHATHPC_FINETUNED_MODEL_PATH="./app/peft_adapter"\
    CHATHPC_MERGED_MODEL_PATH="./app/merged_adapters"\
    CHATHPC_TRAINING_OUTPUT_DIR="./app/kokkos-code-llama"\
    CHATHPC_PROMPT_TEMPLATE="You are a powerful LLM model for Kokkos called ChatKokkos created by ORNL. Your job is to answer questions about the Kokkos programming model. You are given a question and context regarding the Kokkos programming model.\n\nYou must output the answer the question.\n\n### Context:\n{{ context }}\n\n### Question:\n{{ question }}\n\n### Answer:\n{{ answer }}\n\n"\
    chathpc config
CHATHPC_DATA_FILE="/home/7ry/Data/ellora/ChatKokkos-data/kokkos_dataset_before_reinforcement.json"\
    CHATHPC_FINETUNED_MODEL_PATH="./app/peft_adapter"\
    CHATHPC_MERGED_MODEL_PATH="./app/merged_adapters"\
    CHATHPC_TRAINING_OUTPUT_DIR="./app/kokkos-code-llama"\
    CHATHPC_PROMPT_TEMPLATE="You are a powerful LLM model for Kokkos called ChatKokkos created by ORNL. Your job is to answer questions about the Kokkos programming model. You are given a question and context regarding the Kokkos programming model.\n\nYou must output the answer the question.\n\n### Context:\n{{ context }}\n\n### Question:\n{{ question }}\n\n### Answer:\n{{ answer }}\n\n"\
    chathpc config
