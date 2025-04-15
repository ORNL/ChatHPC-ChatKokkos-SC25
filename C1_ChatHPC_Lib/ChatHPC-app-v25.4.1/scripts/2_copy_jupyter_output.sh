#!/usr/bin/env bash
GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cp $GIT_ROOT/examples/fine-tune-chat-kokkos_output.ipynb $GIT_ROOT/examples/fine-tune-chat-kokkos.ipynb
cp $GIT_ROOT/examples/fine-tune-chat-kokkos-app_output.ipynb $GIT_ROOT/examples/fine-tune-chat-kokkos-app.ipynb
