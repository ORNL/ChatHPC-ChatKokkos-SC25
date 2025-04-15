import unittest
from pathlib import Path

from chathpc.app.app import App, AppConfig


def test_chat_prompt_json():
    """Test basice prompt fuction from reading JSON config."""
    app = App.from_json("tests/files/config.json")

    expected = "You are a powerful LLM model for Kokkos called ChatKokkos created by ORNL. Your job is to answer questions about the Kokkos programming model. You are given a question and context regarding the Kokkos programming model.\n\nYou must output the answer the question.\n\n### Context:\nContext\n\n### Question:\nQuestion\n\n### Answer:\n"
    result = app.chat_prompt(question="Question", context="Context")
    assert result == expected, "Prompt is not as expected."


def test_chat_prompt_json_no_context():
    """Test without context"""
    app = App.from_json("tests/files/config.json")

    expected = "You are a powerful LLM model for Kokkos called ChatKokkos created by ORNL. Your job is to answer questions about the Kokkos programming model. You are given a question and context regarding the Kokkos programming model.\n\nYou must output the answer the question.\n\n### Question:\nQuestion\n\n### Answer:\n"
    result = app.chat_prompt(question="Question")
    assert result == expected, "Prompt is not as expected."


def test_training_prompt_json():
    """Test basice prompt fuction from simple template setting."""
    app = App.from_json(
        "tests/files/config.json",
    )
    expected = "You are a powerful LLM model for Kokkos called ChatKokkos created by ORNL. Your job is to answer questions about the Kokkos programming model. You are given a question and context regarding the Kokkos programming model.\n\nYou must output the answer the question.\n\n### Context:\nContext\n\n### Question:\nQuestion\n\n### Answer:\nAnswer\n\n"

    result = app.training_prompt(user="Question", context="Context", assistant="Answer")
    assert result == expected, "Prompt is not as expected."


def test_training_prompt_json2():
    """Test basice prompt fuction from simple template setting."""
    app = App.from_json(
        "tests/files/config.json",
    )
    inputd = {
        "question": "Can you give me an example of a Kokkos view?",
        "context": "Introduction to Kokkos programming model",
        "answer": "```cpp\n#include <Kokkos_Core.hpp>\nint main( int argc, char* argv[] ) {\n  int M = 10;\n  Kokkos::initialize( argc, argv );{\n    Kokkos::View<float *> X(M);\n    Kokkos::View<float **> Y(M, M);\n    Kokkos::kokkos_free<>(X);\n    Kokkos::kokkos_free<>(Y);\n  }\n  Kokkos::finalize();\n  return 0;\n}\n```",
    }
    expected = "You are a powerful LLM model for Kokkos called ChatKokkos created by ORNL. Your job is to answer questions about the Kokkos programming model. You are given a question and context regarding the Kokkos programming model.\n\nYou must output the answer the question.\n\n### Context:\nIntroduction to Kokkos programming model\n\n### Question:\nCan you give me an example of a Kokkos view?\n\n### Answer:\n```cpp\n#include <Kokkos_Core.hpp>\nint main( int argc, char* argv[] ) {\n  int M = 10;\n  Kokkos::initialize( argc, argv );{\n    Kokkos::View<float *> X(M);\n    Kokkos::View<float **> Y(M, M);\n    Kokkos::kokkos_free<>(X);\n    Kokkos::kokkos_free<>(Y);\n  }\n  Kokkos::finalize();\n  return 0;\n}\n```\n\n"

    result = app.training_prompt(**inputd)
    assert result == expected, "Prompt is not as expected."


def test_chat_prompt_simple():
    """Test basice prompt fuction from simple template setting."""
    config = AppConfig.from_json(
        "tests/files/config.json",
        extra_params={
            "prompt_template": "System\n\n### Input:\n{{question}}\n\n### Context:\n{{context}}\n\n### Response:\n{{answer}}\n",
            "prompt_template_file": None,
        },
    )
    app = App(config)
    expected = "System\n\n### Input:\nQuestion\n\n### Context:\nContext\n\n### Response:\n"
    result = app.chat_prompt(prompt="Question", context="Context")
    assert result == expected, "Prompt is not as expected."


def test_training_prompt_simple():
    """Test basice prompt fuction from simple template setting."""
    app = App.from_json(
        "tests/files/config.json",
        extra_params={
            "prompt_template": "System\n\n### Input:\n{{question}}\n\n### Context:\n{{context}}\n\n### Response:\n{{answer}}\n\n",
            "prompt_template_file": None,
        },
    )
    expected = "System\n\n### Input:\nQuestion\n\n### Context:\nContext\n\n### Response:\nAnswer\n\n"
    result = app.training_prompt(user="Question", context="Context", assistant="Answer")
    assert result == expected, "Prompt is not as expected."


def test_training_prompt_liternal_newline():
    """Test literal newline in template."""
    app = App.from_json(
        "tests/files/config.json",
        extra_params={
            "prompt_template": "System\\n\\n### Input:\\n{{question}}\\n\\n### Context:\\n{{context}}\\n\\n### Response:\\n{{answer}}\\n\\n",
            "prompt_template_file": None,
        },
    )
    expected = "System\n\n### Input:\nQuestion\n\n### Context:\nContext\n\n### Response:\nAnswer\n\n"
    result = app.training_prompt(user="Question", context="Context", assistant="Answer")
    assert result == expected, "Prompt is not as expected."


def test_save_readme():
    app = App.from_json(
        "tests/files/config.json",
    )
    app.save_readme("tests/files/README.md")
    assert Path("tests/files/README.md").is_file()
    Path("tests/files/README.md").unlink()


def test_save_readme_folder():
    app = App.from_json(
        "tests/files/config.json",
    )
    app.save_readme("tests/files")
    assert Path("tests/files/README.md").is_file()
    Path("tests/files/README.md").unlink()


if __name__ == "__main__":
    unittest.main()
