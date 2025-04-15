#!/usr/bin/env python3

"""Extract example output from jupyter notebook."""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import traceback
from subprocess import check_output

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


def check_words_in_string(words, string):
    return any(word in string for word in words)


def extract_output(file: str):
    # Load the notebook
    with open(file) as f:
        notebook = json.load(f)

    # Iterate through the cells and print outputs
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            generate = False

            for i in cell.get("source", []):
                if check_words_in_string(
                    ["model.generate", "_evaluate", "chat_app.tokenize_training_set()", "generate_and_tokenize_prompt"],
                    i,
                ):
                    generate = True
                    break

            if generate:
                for output in cell.get("outputs", []):
                    if "text" in output:
                        print("".join(output["text"]))
                    elif "data" in output:
                        for data in output["data"].values():
                            print(data)


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

    for file in args.files:
        extract_output(file)


if __name__ == "__main__":
    main()
