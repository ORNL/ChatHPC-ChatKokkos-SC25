#!/usr/bin/env python3

"""Uses the output from scripts/test_training.sh to verify the the models trained properly."""

import argparse
import contextlib
import os
import subprocess
import sys
import traceback
from collections import OrderedDict
from functools import partial
from subprocess import check_output

from tqdm import tqdm

from chathpc.app.utils.datastore import read_or_new_json
from chathpc.app.utils.template_utils import map_keywords
from chathpc.app.utils.verify_utils import ignore_minor

GIT_ROOT = check_output("git rev-parse --show-toplevel", shell=True).decode().strip()  # noqa S602
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def run(command, verbose=True, noop=False, directory=None):
    """Print command then run command"""
    return_val = ""

    if directory is not None:
        with pushd(directory):
            return run(command, verbose, noop)

    if verbose:
        print(command)
    if not noop:
        try:
            return_val = subprocess.check_output(command, shell=True, stderr=subprocess.PIPE).decode()  # noqa: S602
        except subprocess.CalledProcessError as e:
            err_mesg = f"{os.getcwd()}: {e}\n\n{traceback.format_exc()}\n\n{e.returncode}\n\n{e.stdout.decode()}\n\n{e.stderr.decode()}"
            print(err_mesg, file=sys.stderr)
            with open("err.txt", "w") as fd:
                fd.write(err_mesg)
            raise
        except Exception as e:
            err_mesg = f"{os.getcwd()}: {e}\n\n{traceback.format_exc()}"
            print(err_mesg, file=sys.stderr)
            with open("err.txt", "w") as fd:
                fd.write(err_mesg)
            raise
        if verbose and return_val:
            print(return_val)

    return return_val


def shell_source(script):
    """Sometime you want to emulate the action of "source" in bash,
    settings some environment variables. Here is a way to do it."""
    import os
    import subprocess

    pipe = subprocess.Popen(f"bash -c 'source {script} > /dev/null; env'", stdout=subprocess.PIPE, shell=True)  # noqa: S602
    output = pipe.communicate()[0].decode()
    env = dict(line.split("=", 1) for line in output.splitlines())
    os.environ.update(env)


def run_experiment_chat_app(experiment, basepath, template):
    from chathpc.app import App as ChatApp

    chat_app = ChatApp.from_json(
        {
            "prompt_template": template,
            "finetuned_model_path": f"{basepath}/peft_adapter",
            "merged_model_path": f"{basepath}/merged_adapters",
            "training_output_dir": f"{basepath}/kokkos-code-llama",
        }
    )
    chat_app.load_datasets()

    def get_finetuned():
        chat_app.load_finetuned_model()
        finetune = []
        for i, item in tqdm(enumerate(chat_app.train_dataset), "Run Finetune", total=len(chat_app.train_dataset)):  # type: ignore
            response = chat_app.chat_evaluate_extract(**item)
            prompt = chat_app.chat_prompt(**item)
            training_prompt = chat_app.training_prompt(**item)
            datapoint = OrderedDict(
                [
                    ("index", i),
                    ("prompt", prompt),
                    ("training_prompt", training_prompt),
                    ("question", item["question"]),
                    ("context", item["context"]),
                    ("answer", item["answer"]),
                    ("response", response),
                ]
            )
            finetune.append(datapoint)
        return finetune

    finetune = read_or_new_json(f"{experiment}_finetune_out", get_finetuned)

    def get_merged():
        chat_app.load_merged_model()
        merged = []
        for i, item in tqdm(enumerate(chat_app.train_dataset), "Run Merged", total=len(chat_app.train_dataset)):  # type: ignore
            response = chat_app.chat_evaluate_extract(**item)
            prompt = chat_app.chat_prompt(**item)
            training_prompt = chat_app.training_prompt(**item)
            datapoint = OrderedDict(
                [
                    ("index", i),
                    ("prompt", prompt),
                    ("training_prompt", training_prompt),
                    ("question", item["question"]),
                    ("context", item["context"]),
                    ("answer", item["answer"]),
                    ("response", response),
                ]
            )
            merged.append(datapoint)
        return merged

    merged = read_or_new_json(f"{experiment}_merged_out", get_merged)

    return (finetune, merged)


def run_ollama(template):
    from ollama import GenerateResponse, generate

    from chathpc.app import App as ChatApp

    experiment = "ollama"
    chat_app = ChatApp.from_json({"prompt_template": template})
    chat_app.load_datasets()

    def get_ol():
        ol = []
        for i, item in tqdm(enumerate(chat_app.train_dataset), "Run ol", total=len(chat_app.train_dataset)):  # type: ignore
            response: GenerateResponse = generate(
                model="ChatKokkos",
                prompt=map_keywords(item)["prompt"],
                system=item["context"],
                options={"temperature": 0.0},
            )
            datapoint = OrderedDict(
                [
                    ("index", i),
                    ("question", item["question"]),
                    ("context", item["context"]),
                    ("answer", item["answer"]),
                    ("response", response.response.strip()),
                ]
            )
            ol.append(datapoint)
        return ol

    return read_or_new_json(f"{experiment}_ol_out", get_ol)


def run_ollama_chat(template):
    from ollama import ChatResponse, chat

    from chathpc.app import App as ChatApp

    experiment = "ollama"
    chat_app = ChatApp.from_json({"prompt_template": template})
    chat_app.load_datasets()

    def get_ol_chat():
        ol = []
        for i, item in tqdm(enumerate(chat_app.train_dataset), "Run ol Chat", total=len(chat_app.train_dataset)):  # type: ignore
            response: ChatResponse = chat(
                model="ChatKokkos",
                options={"temperature": 0.0},
                messages=[
                    {"role": "system", "content": item["context"]},
                    {"role": "user", "content": map_keywords(item)["prompt"]},
                ],
            )
            training_prompt = chat_app.training_prompt(**item)
            datapoint = OrderedDict(
                [
                    ("index", i),
                    ("training_prompt", training_prompt),
                    ("question", item["question"]),
                    ("context", item["context"]),
                    ("answer", item["answer"]),
                    ("response", response.message.content),
                ]
            )
            ol.append(datapoint)
        return ol

    return read_or_new_json(f"{experiment}_ol_chat_out", get_ol_chat)


def verify_app(runner):
    (finetuned, merged) = runner()

    response_errors = 0
    merge_errors = 0

    for i, (fine, merge) in tqdm(enumerate(zip(finetuned, merged)), "Compare"):
        if fine["answer"] != merge["answer"]:
            print("Error: answer mismatch")
            print(f"Sample {i}")
            print(f"Finetuned:\n{fine['answer']}")
            print(f"Merged:\n{merge['answer']}")
            print("**********************************************************")
            print()
            raise RuntimeError("Answer Mismatch")
        if ignore_minor(fine["answer"]) != ignore_minor(fine["response"]):
            response_errors += 1
            print("Error: response mismatch")
            print(f"Sample {i}")
            print(f"Answer:\n{fine['answer']}")
            print(f"Response:\n{fine['response']}")
            print("**********************************************************")
            print()
        if ignore_minor(fine["response"]) != ignore_minor(merge["response"]):
            merge_errors += 1
            print("Error: merge mismatch")
            print(f"Sample {i}")
            print(f"Finetuned:\n{fine['response']}")
            print(f"Merged:\n{merge['response']}")
            print("**********************************************************")
            print()

    print(f"Response Errors: {response_errors}, Merge Errors: {merge_errors}")
    return response_errors, merge_errors


def verify_ollama(template):
    ol = run_ollama(template=template)
    ol_chat = run_ollama_chat(template=template)

    response_errors = 0
    ol_errors = 0
    olc_errors = 0

    for i, (o, oc) in tqdm(enumerate(zip(ol, ol_chat)), "Compare"):  # type: ignore
        if o["answer"] != oc["answer"]:
            print("Error: answer mismatch")
            print(f"Sample {i}")
            print(f"Ollama:\n{o['answer']}")
            print(f"Ollama Chat:\n{oc['answer']}")
            print("**********************************************************")
            print()
            raise RuntimeError("Answer Mismatch")
        if ignore_minor(o["answer"]) != ignore_minor(o["response"]):
            response_errors += 1
            # print("Error: response mismatch")
            # print(f"Sample {i}")
            # print(f"Answer:\n{o['answer']}")
            # print(f"Response:\n{o['response']}")
            # print(f"**********************************************************")
            # print()
        if ignore_minor(o["answer"]) != ignore_minor(o["response"]):
            ol_errors += 1
            print("Error: ollama mismatch")
            print(f"Sample {i}")
            print(f"Answer:\n{o['answer']}")
            print(f"Ollama:\n{o['response']}")
            print("**********************************************************")
            print()
        if ignore_minor(o["response"]) != ignore_minor(oc["response"]):
            olc_errors += 1
            print("Error: ollama chat mismatch")
            print(f"Sample {i}")
            print(f"Ollama:\n{o['response']}")
            print(f"Ollama Chat:\n{oc['response']}")
            print("**********************************************************")
            print()

    print(f"Ollama Errors: {ol_errors}")
    return ol_errors, olc_errors


def init_parser(parser):
    # parser.add_argument('-d', '--dir', type=str, default=OUTPUT_DIR)
    parser.add_argument("--debug", action="store_true", help="Open debug port (5678).")
    parser.add_argument("files", metavar="p", type=str, nargs="*")


def main(raw_args=None):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Extract example output from jupyter notebook.""")
    init_parser(parser)
    args = parser.parse_args(raw_args)

    if args.debug:
        import debugpy  # noqa: T100

        debugpy.listen(5678)  # noqa: T100
        print("Attach debugger to continue.")
        debugpy.wait_for_client()  # noqa: T100

    os.environ["CHATHPC_DATA_FILE"] = "/home/7ry/Data/ellora/ChatKokkos-data/kokkos_dataset_before_reinforcement.json"

    old_template = "You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.\n\nYou must output the Kokkos question that answers the question.\n\n### Input:\n{{question}}\n\n### Context:\n{{context}}\n\n### Response:\n{{answer}}\n"

    new_template = "You are a powerful LLM model for Kokkos called ChatKokkos created by ORNL. Your job is to answer questions about the Kokkos programming model. You are given a question and context regarding the Kokkos programming model.\n\nYou must output the answer the question.\n\n### Context:\n{{ context }}\n\n### Question:\n{{ question }}\n\n### Answer:\n{{ answer }}\n\n"

    # Notebook
    print("** Running Notebook **")
    print("** Running Notebook **", file=sys.stderr)
    notebook_errors = verify_app(
        partial(run_experiment_chat_app, experiment="jupyter", basepath=".", template=new_template)
    )
    print("Response Errors: {}, Merge Errors: {}".format(*notebook_errors))
    print("Response Errors: {}, Merge Errors: {}".format(*notebook_errors), file=sys.stderr)

    # Notebook App
    print("\n\n** Running Notebook App **")
    print("\n\n** Running Notebook App **", file=sys.stderr)
    notebook_app_errors = verify_app(
        partial(run_experiment_chat_app, experiment="jupyter_app", basepath="./jupyter_app", template=new_template)
    )
    print("Response Errors: {}, Merge Errors: {}".format(*notebook_app_errors))
    print("Response Errors: {}, Merge Errors: {}".format(*notebook_app_errors), file=sys.stderr)

    # App New
    print("\n\n** Running App New **")
    print("\n\n** Running App New **", file=sys.stderr)
    app_new_errors = verify_app(
        partial(run_experiment_chat_app, experiment="app_new", basepath="../", template=new_template)
    )
    print("Response Errors: {}, Merge Errors: {}".format(*app_new_errors))
    print("Response Errors: {}, Merge Errors: {}".format(*app_new_errors), file=sys.stderr)

    # App
    print("\n\n** Running App **")
    print("\n\n** Running App **", file=sys.stderr)
    app_errors = verify_app(partial(run_experiment_chat_app, experiment="app", basepath="./app", template=new_template))
    print("Response Errors: {}, Merge Errors: {}".format(*app_errors))
    print("Response Errors: {}, Merge Errors: {}".format(*app_errors), file=sys.stderr)

    # App Old
    print("\n\n** Running App Old **")
    print("\n\n** Running App Old **", file=sys.stderr)
    app_old_errors = verify_app(
        partial(run_experiment_chat_app, experiment="app_old", basepath="./app_old", template=old_template)
    )
    print("Response Errors: {}, Merge Errors: {}".format(*app_old_errors))
    print("Response Errors: {}, Merge Errors: {}".format(*app_old_errors), file=sys.stderr)

    # # App Prior
    # print("\n\n** Running App Prior **")
    # print("\n\n** Running App Prior **", file=sys.stderr)
    # app_prior_errors = verify_app(
    #     partial(
    #         run_experiment_chat_app,
    #         experiment="app_prior",
    #         basepath="/home/7ry/Data/ellora/ChatHPC-app-main/examples/app",
    #         template=old_template,
    #     )
    # )
    # print("Response Errors: {}, Merge Errors: {}".format(*app_prior_errors))
    # print("Response Errors: {}, Merge Errors: {}".format(*app_prior_errors), file=sys.stderr)

    # Ollama
    print("\n\n** Running Ollama **")
    print("\n\n** Running Ollama **", file=sys.stderr)
    ol_errors = verify_ollama(template=new_template)
    print("Ollama Errors: {}, Ollama Chat Errors: {}".format(*ol_errors))
    print("Ollama Errors: {}, Ollama Chat Errors: {}".format(*ol_errors), file=sys.stderr)


if __name__ == "__main__":
    main()
