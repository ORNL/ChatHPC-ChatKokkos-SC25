# ChatHPC Application

Documentation: <https://devdocs.ornl.gov/ChatHPC/ChatHPC-app>

Coverage Report: <https://devdocs.ornl.gov/ChatHPC/ChatHPC-app/coverage>

**Table of Contents**
- [ChatHPC Application](#chathpc-application)
    - [Installation](#installation)
    - [Setup pre-commit Git hooks](#setup-pre-commit-git-hooks)
    - [Quick Start](#quick-start)
    - [CLI Interface](#cli-interface)
        - [ChatHPC](#chathpc)
        - [ChatHPC JSON to MD](#chathpc-json-to-md)
    - [Running with hatch](#running-with-hatch)
    - [Testing with hatch](#testing-with-hatch)
    - [Format code with hatch](#format-code-with-hatch)
    - [View version with hatch](#view-version-with-hatch)
    - [Update version with hatch](#update-version-with-hatch)
    - [Update version with script](#update-version-with-script)
    - [Documentation](#documentation)
        - [Commands](#commands)
        - [Hatch Commands](#hatch-commands)

## Installation

For development in folder:

```bash
git clone git@code.ornl.gov:ChatHPC/ChatHPC-app.git
cd ChatHPC-app
python3 -m venv --upgrade-deps --prompt $(basename $PWD) .venv
source .venv/bin/activate
pip install -e .
```

For use in virtual environment:

```bash
python3 -m venv --upgrade-deps --prompt $(basename $PWD) .venv
source .venv/bin/activate
pip install git+ssh://git@code.ornl.gov/ChatHPC/ChatHPC-app.git
```

## Setup pre-commit Git hooks

Use hatch or install pre-commit inside python virtual environment.
```bash
hatch shell
```
or
```bash
pip install pre-commit
```

Then install the hooks.
```bash
pre-commit install
```

Note: You might have to upgrade pre-commit.
```bash
pre-commit autoupdate
```

Note: The markdown linter requires Ruby gem to be installed to auto-install and run mdl.

On Ubuntu this can be done with:
```bash
sudo apt install ruby-full
```

## Quick Start

See [Creating a new ChatHPC application.](https://devdocs.ornl.gov/ChatHPC/ChatHPC-project/#how-to-create-a-new-chathpcchatx-application-repo).

## CLI Interface

### ChatHPC

Get Help:
```bash
$ chathpc --help
Usage: chathpc [OPTIONS] COMMAND [ARGS]...

Options:
  -h, --help  Show this message and exit.

Commands:
  config      Print current config
  run         Interact with the model.
  run-base    Interact with the base model.
  run-fine    Interact with the finetuned model.
  run-merged  Interact with the merged model.
  train       Finetune the model.
```

Run interactively:
```bash
chathpc run
```

Example interactive session:
```bash
$ chathpc run
chathpc ()> /context
Context: Introduction to Kokkos programming model
chathpc (Introduction to Kokkos programming model)> Which kind of Kokkos views are?
<s> You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.

You must output the Kokkos question that answers the question.

### Input:
Which kind of Kokkos views are?

### Context:
Introduction to Kokkos programming model

### Response:
There are two different layouts; LayoutLeft and LayoutRight.
</s>
chathpc (Introduction to Kokkos programming model)> \bye
```

Train:
```bash
chathpc train
```

### ChatHPC JSON to MD

Get Help:
```shell
$ chathpc-json-to-md -h
usage: chathpc-json-to-md [-h] [--debug] [--log_level LOG_LEVEL] [--add_rating_template] [json]

Convert Json files to Markdown for ease of reading.

positional arguments:
  json                  Json string or path to json file.

options:
  -h, --help            show this help message and exit
  --debug               Open debug port (5678).
  --log_level LOG_LEVEL
                        Log level.
  --add_rating_template
                        Add rating template to markdown.

```

Example:
```shell
chathpc-json-to-md input.json > output.md
```

## Running with hatch

```bash
hatch shell
```

## Testing with hatch

```bash
hatch run test
```

To test on all python versions:
```bash
hatch run all:test
```

Run tests and print the output.
```bash
hatch run test -v -s
```

## Format code with hatch

```bash
hatch fmt
```

Update default ruff rules:

```bash
hatch fmt --check --sync
```

## View version with hatch

```bash
hatch version
```

## Update version with hatch

```bash
hatch version <new version>
```

## Update version with script

An automated script is provided to update the version using a date based version. This scripts will determine the next version to use and then update the version, update the changelog, and commit the changes. Lastly, it will tag the commit.

```bash
scripts/version_bump.py
```

## Documentation

Documentation is built with [mkdocs](https://www.mkdocs.org) using the [Read the Docs](https://docs.readthedocs.io/en/stable/) theme.

### Commands

- `mkdocs new [dir-name]` - Create a new project.
- `mkdocs serve` - Start the live-reloading docs server.
- `mkdocs build` - Build the documentation site.
- `mkdocs -h` - Print help message and exit.

Other useful commands:
- `mkdocs serve -a 0.0.0.0:8000` - Serve with extenal access to the site. (Useful in ExCL to view using foxyproxy.)

### Hatch Commands

View environment
```bash
hatch env show docs
```

Build documentation
```bash
hatch run docs:build
```

Serve documentation
```bash
hatch run docs:serve
```
or
```bash
hatch run docs:serve -a 0.0.0.0:8000
```
