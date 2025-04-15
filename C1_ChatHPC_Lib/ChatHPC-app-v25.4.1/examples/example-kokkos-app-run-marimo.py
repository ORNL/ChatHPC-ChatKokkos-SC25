# ruff: noqa: S101, INP001

import marimo

__generated_with = "0.11.8"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Fine-tunning ChatKokkos Example""")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        These are the steps taken to fine-tune ChatKokkos. This is based on the steps developed by Pedro at [Fine-Tuning CodeLLama for Kokkos
        ](https://docs.google.com/document/d/1u_r9PKUYYV_n5vte4oHDeZiPjUa_hnCS-pqdoB8YmF4/edit?tab=t.0) and on the [Hugging Face PEFT Adaptor Training Guide](https://huggingface.co/docs/transformers/en/peft).
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Libraries""")


@app.cell
def _(logger_setup):
    assert logger_setup
    import os

    from chathpc.app import App as ChatApp

    chat_app = ChatApp.from_json(
        "./tests/files/config.json",
        {
            "finetuned_model_path": "./peft_adapter",
            "merged_model_path": "./merged_adapters",
            "training_output_dir": "./kokkos-code-llama",
        },
    )

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    chat_app.print_config()
    return ChatApp, chat_app, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Model""")


@app.cell
def _(chat_app):
    chat_app.load_base_model()
    base_model_loaded = True
    return (base_model_loaded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Test base model""")


@app.cell
def _(base_model_loaded, chat_app):
    assert base_model_loaded

    question = "Which kind of Kokkos views are?"
    context = "Introduction to Kokkos programming model"

    ### Response:
    output = chat_app.chat_evaluate(question=question, context=context, max_new_tokens=200)
    print(output)
    return context, output, question


@app.cell
def _(base_model_loaded, chat_app):
    assert base_model_loaded

    question_1 = "Which compilers can I use to compile Kokkos codes?"
    context_1 = "Kokkos installation"
    output_1 = chat_app.chat_evaluate(question=question_1, context=context_1, max_new_tokens=200)
    print(output_1)
    return context_1, output_1, question_1


@app.cell
def _(base_model_loaded, chat_app):
    assert base_model_loaded

    question_2 = "Can you give me an example of Kokkos parallel_reduce?"
    context_2 = "Introduction to Kokkos programming model"
    output_2 = chat_app.chat_evaluate(question=question_2, context=context_2, max_new_tokens=200)
    print(output_2)
    return context_2, output_2, question_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Tokenization""")


@app.cell
def _(chat_app):
    chat_app.load_datasets()
    chat_app.tokenize_training_set()
    print(len(chat_app.tokenized_train_dataset))
    print(chat_app.tokenized_train_dataset[0])
    print(chat_app.tokenized_train_dataset[1])
    print(chat_app.tokenized_train_dataset[2])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Trained Model""")


@app.cell
def _(chat_app):
    chat_app.load_finetuned_model()
    finetuned_model_loaded = True
    return (finetuned_model_loaded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Evaluate Trained Model""")


@app.cell
def _(chat_app, finetuned_model_loaded):
    assert finetuned_model_loaded

    question_3 = "Which kind of Kokkos views are?"
    context_3 = "Introduction to Kokkos programming model"
    output_3 = chat_app.chat_evaluate(question=question_3, context=context_3, max_new_tokens=500)
    print(output_3)
    return context_3, output_3, question_3


@app.cell
def _(chat_app, finetuned_model_loaded):
    assert finetuned_model_loaded

    question_4 = "Which compilers can I use to compile Kokkos codes?"
    context_4 = "Kokkos installation"
    output_4 = chat_app.chat_evaluate(question=question_4, context=context_4, max_new_tokens=500)
    print(output_4)
    return context_4, output_4, question_4


@app.cell
def _(chat_app, finetuned_model_loaded):
    assert finetuned_model_loaded

    question_5 = "Can you give me an example of Kokkos parallel_reduce?"
    context_5 = "Introduction to Kokkos programming model"
    output_5 = chat_app.chat_evaluate(question=question_5, context=context_5, max_new_tokens=500)
    print(output_5)
    return context_5, output_5, question_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Appendix""")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Setup Logger""")


@app.cell
def _():
    import sys

    from loguru import logger

    # Remove the default logger and add a new logger at INFO level.
    logger.remove()  # Remove the default handler.
    logger.add(sys.stderr, level="INFO")

    logger_setup = True
    return logger, logger_setup, sys


if __name__ == "__main__":
    app.run()
