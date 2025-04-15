# ChatHPC Application

See documentation at <https://devdocs.ornl.gov/ChatHPC/ChatHPC-project> for general usage information.

## Tools Used

- [Hatch](https://hatch.pypa.io/) --- Python Build System.
- [MkDocs](https://www.mkdocs.org/) --- Documentation Generator.
    - [Material Theme](https://squidfunk.github.io/mkdocs-material/) --- Theme for documentation.
    - [mkdocstrings](https://mkdocstrings.github.io/) --- Automatic documentation generation from sources.
    - [DevDocs](https://docs.excl.ornl.gov/quick-start-guides/devdocs) --- Internal to ORNL document website hosting.
- [GitLab CI](https://docs.gitlab.com/ee/ci/) --- Continuous Integration.
    - [Example Pipeline](https://code.ornl.gov/ChatHPC/ChatHPC-app/-/pipelines)
    - [Example Pipeline Source](https://code.ornl.gov/ChatHPC/ChatHPC-app/-/blob/main/.gitlab-ci.yml?ref_type=heads)
- [Ruff](https://docs.astral.sh/ruff/) --- Python linter and code formater.
    - [Ruff Rules](https://docs.astral.sh/ruff/rules/) --- Rules used by Ruff.
- [EditorConfig](https://editorconfig.org/) --- Maintain consistent coding styles between different editors and IDEs.
- [Markdown Lint Tool](https://github.com/markdownlint/markdownlint) --- Markdown linting tool.
- [Pre-Commit](https://pre-commit.com/) --- Git precommit hooks.
    - [Built-in Hooks](https://github.com/pre-commit/pre-commit-hooks)
    - [Ruff Pre-Commit Hooks](https://github.com/astral-sh/ruff-pre-commit)
    - [Editor Config Pre-Commit Hooks](https://github.com/editorconfig-checker/editorconfig-checker.python)
    - [Markdown Lint Pre-Commit Hooks](https://github.com/markdownlint/markdownlint)
