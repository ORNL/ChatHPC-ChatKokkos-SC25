# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to a custom Development Versioning specified by Aaron Young.

A summary of Development Versioning Specification is shown below.

> Given a version number BRANCH.TAG.BUILD, increment the:
> 1. BRANCH version when you make breaking/major changes that you want to track in a separate branch.
> 2. TAG version when you make a new tag to mark a specific spot.
> 3. BUILD version when you create a new build with artifacts or bug fixes for that you want to point to.
>
> Then for your repo you have branch versions for each version. For example branches v0 and v1. Then when you create tags, say on branch v0, you would create tags v0.0.0, v0.1.0, and v0.2.0.
> CI or a manual process could add v0.0.x branches as new changes are added to a local branch. BUILD is also used when patches are applied to a tagged branch, after the patch is applied, add a new tag with BUILD + 1.
>
> `main` always points to the current major branch plus 1. `dev` is an integration branch before merging into `main`. When `dev` is merged into `main`, the TAG is updated.

An alternative approach is to use date-based versioning.

With this method, the version is YEAR.MONTH.RELEASE. To increment this version, use the year and the date without 0 padding for the first two numbers. I prefer to use the year without the centary. Then increment the RELEASE number to a unique release. This process is done automatically by the `scripts/version_bump.py` script. Using this script is the prefered method for versioning without planned backporting of fixes.

## [Unreleased]

## [25.4.1] - 2025-04-15

### Fixed

- Found various small bugs with added YAML support.

## [25.4.0] - 2025-04-15

This release adds support for YAML files as both data files and config files as the main feature.

### Added

- CLI: Added `chathpc-data-to-md` to convert a json or yaml output file to markdown for easier reading (alias of `chathpc-json-to-md`).
- CLI: Added `chathpc-json-to-yaml` to convert a json file to yaml for easier editing.

### Changed

- Changed `load_json_arg` to `load_json_yaml_arg` with added yaml parsing support.
- Changed depenency: readline -> gnureadline.

### Fixed

- Fixed item keyword mapping in test and verify.
- Fix edge cases with generating markdown from the json.

## [25.3.0] - 2025-03-04

### Added

- CI: Test command to evalute all the samples in a json file.
- CI: Ollama subcommands to run verify and test using an Ollama model.
- CI: OpenAI subcommands to run verify and test using an OpenAI model.
- CI: Base subcommands to run verify and test using the base model.
- Ollama module: Added to make evaluating Ollama models easier.
- OpenAI module: Added to make evaluating OpenAI models easier.
- Training Tokenizer: Added training input cropping warning.
- CLI: Added `chathpc-json-to-md` to convert a json or yaml output file to markdown for easier reading (same as `chathpc-data-to-md`).

### Fixed

- Prompt Template: Missing `:`.

## [25.2.1] - 2025-02-26

### Added

- Config: `prompt_template_file` option added. This allows defining a prompt template from a file. If relative path, first the CWD is checked, then the path relative to the config.json.
- Config: `prompt_template` option added. This is a unified prompt template for both training and inference. Replaces `training_prompt` and `inference_prompt`.
- APP: `chat_evaluate_extract` method added to chathpc to evaluate and extract the answer portion in one call.
- APP: `save_readme` function to save a starter readme for models.
- Train: Now saves a template readme in the output folders.
- CLI app: log_level is now a command line argument.
- Interactive: Added optional ability to extract answer from response with `--extract`.
- Interactive: Gracefully handle EOF.
- Interactive & Template: Make context optional.
- Interactive: Allow context to be added inline with `/context <context>`.
- Interactive: A blank context, unsets the context.
- Ollama: Now tests both generate and chat API. It is recommended to use the chat API to be more compatible with OpenAI's api.
- CLI & Method: `Verify` subcommand to verify the model against the training dataset.
- Examples: Added [Marimo](https://marimo.io/) example using ChatHPC.
- Full Test: Top level `scripts/full_test.sh` to run all the tests in order.
- Data Explore: Data explore script `scripts/tests/data_expore.py` to print out an inspect the training dataset.
- Added version to package for easy query of ChatHPC version.

### Changed

- Template: Switch from format string to Jinja for the templates. The main change required here, is to use `{{}}` for variable names instead of `{}`.
- Template: Now only one template is used for both training and inference.
- Template: Can either be set by a file path or a string.
- datastore: Removed dependency on datastore, by coping datastore into utilities.
- verify_app.py: Refactored to remove duplicate code.

## Removed

- Removed `training_prompt`. Replaced by `prompt_template`.
- Removed `inference_prompt`. Replaced by `prompt_template`.

## [25.2.0] - 2025-02-21

### Added

- Added `max_training_tokens` to specify the `max_length` parameter for the tokenizer when tokenizing the training set. Defaults to the prior setting of 512.

### Changed

- Logging: Switch from using logging module to loguru.

### Fixed

- load_json_arg(): Fixed bug with json string as input.
- Fixed bug in interactive run mode.
- Fixed extract_answer() utility to use the prompt and end string to extract the answer portion.

## [25.1.0] - 2025-01-25

Verified initial working version of the ChatHPC App.

### Added

- Initial version of ChatHPC App. Created from the working ChatKokkos Example.

[unreleased]: https://code.ornl.gov/ChatHPC/ChatHPC-app/-/compare/v25.4.1...main
[25.4.1]: https://code.ornl.gov/ChatHPC/ChatHPC-app/-/compare/v25.4.0...v25.4.1
[25.4.0]: https://code.ornl.gov/ChatHPC/ChatHPC-app/-/compare/v25.3.0...v25.4.0
[25.3.0]: https://code.ornl.gov/ChatHPC/ChatHPC-app/-/compare/v25.2.1...v25.3.0
[25.2.1]: https://code.ornl.gov/ChatHPC/ChatHPC-app/-/compare/v25.2.0...v25.2.1
[25.2.0]: https://code.ornl.gov/ChatHPC/ChatHPC-app/-/compare/v25.1.0...v25.2.0
[25.1.0]: https://code.ornl.gov/ChatHPC/ChatHPC-app/-/releases/v25.1.0
