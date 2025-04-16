#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo '*** Running 1_setup.sh ***'
echo
echo '*** Ensure commands are running in correct directory. ***'
echo cd $SCRIPT_DIR
cd $SCRIPT_DIR
echo
echo '*** Check for UV ***'
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "please install uv following https://docs.astral.sh/uv/getting-started/installation/."
    echo "or by running:"
    echo "`curl -LsSf https://astral.sh/uv/install.sh | sh`"
    exit 1
else
    echo "uv is installed."
fi
echo
echo '*** Test running ChatHPC CLI command. ***'
echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc -h
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.4.1 chathpc -h

echo
echo '*** Check for base model weights... ***'
if [ -d "basemodels/CodeLlama-7b-hf" ]; then
    echo "CodeLlama-7b-hf base model found"
else
    echo "CodeLlama-7b-hf base model not found, trying to download from hugging face."
    echo "Note: If this fails, please register your SSH key with Hugging Face, and request access to https://huggingface.co/meta-llama/CodeLlama-7b-hf."
    echo "Alternatively, you can manually download the CodeLlama-7b-hf model from the hugging face website and place it in the basemodels directory."
    echo
    echo git clone git@hf.co:meta-llama/CodeLlama-7b-hf basemodels/CodeLlama-7b-hf
    git clone 'git@hf.co:meta-llama/CodeLlama-7b-hf' 'basemodels/CodeLlama-7b-hf'
    if [ -d "basemodels/CodeLlama-7b-hf" ]; then
        echo "CodeLlama-7b-hf base model found"
    else
        echo "Error: Unable to automatically download CodeLlama-7b-hf base model."
        echo "Please correct errors or manually download and place in basemodels directory."
        exit 1
    fi
fi