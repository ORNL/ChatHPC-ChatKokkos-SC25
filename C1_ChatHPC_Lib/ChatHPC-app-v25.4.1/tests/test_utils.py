import json
import unittest
from pathlib import Path

import pytest

from chathpc.app.app import App
from chathpc.app.json_to_yaml import json_to_yaml
from chathpc.app.utils.common_utils import load_json_yaml_arg


class TestLoadJson(unittest.TestCase):
    def test_json_dict(self):
        j = load_json_yaml_arg({"name": "test"})
        assert j == {"name": "test"}

    def test_json_str(self):
        j = load_json_yaml_arg('{"name": "test"}')
        assert j == {"name": "test"}

    def test_json_none(self):
        j = load_json_yaml_arg(None)
        assert j == {}

    def test_json_file(self):
        filename = "tests/files/config.json"
        j = load_json_yaml_arg(filename)
        with open(filename) as f:
            jj = json.loads(f.read())
            jj["filename"] = filename
        assert j == jj

    def test_json_path(self):
        path = Path("tests/files/config.json")
        j = load_json_yaml_arg(path)
        with open(path) as f:
            jj = json.loads(f.read())
            jj["filename"] = path
        assert j == jj

    def test_bad_file(self):
        path = Path("tests/files/_bad_config.json")
        with pytest.raises(ValueError, match="Invalid JSON or YAML string or missing file:"):
            load_json_yaml_arg(path)


class TestYAML(unittest.TestCase):
    def test_convert_to_yaml(self):
        filename = "tests/files/config.json"
        j = load_json_yaml_arg(filename)
        yaml_str = json_to_yaml(j)
        with open("tests/files/config.yaml", "w") as fd:
            fd.write(yaml_str)

    @pytest.mark.depends(on=["TestYAML::test_convert_to_yaml"])
    def test_yaml_file(self):
        filename = "tests/files/config.json"
        filename_yaml = "tests/files/config.yaml"
        j = load_json_yaml_arg(filename_yaml)
        with open(filename) as f:
            jj = json.loads(f.read())
            jj["filename"] = filename_yaml
        assert j == jj

    @pytest.mark.depends(on=["TestYAML::test_convert_to_yaml"])
    def test_yaml_path(self):
        path = Path("tests/files/config.json")
        path_yaml = "tests/files/config.yaml"
        j = load_json_yaml_arg(path_yaml)
        with open(path) as f:
            jj = json.loads(f.read())
            jj["filename"] = path_yaml
        assert j == jj

    def test_yaml_data1(self):
        j = load_json_yaml_arg("tests/files/data-test.json", False)
        jj = load_json_yaml_arg("tests/files/data-test1.yaml", False)
        assert j == jj

    def test_yaml_data2(self):
        j = load_json_yaml_arg("tests/files/data-test.json", False)
        jj = load_json_yaml_arg("tests/files/data-test2.yaml", False)
        assert j == jj

    def test_yaml_data3(self):
        j = load_json_yaml_arg("tests/files/data-test.json", False)
        yaml_str = json_to_yaml(j)
        with open("tests/files/data-test3.yaml", "w") as fd:
            fd.write(yaml_str)
        jj = load_json_yaml_arg("tests/files/data-test3.yaml", False)
        assert j == jj


class TestExtractAnswer(unittest.TestCase):
    def test_extract_answer_simple(self):
        app = App.from_json(
            "tests/files/config.json",
            extra_params={
                "prompt_template": "System\n\n### Input:\n{{question}}\n\n### Context:\n{{context}}\n\n### Response:\n{{answer}}\n\n",
                "prompt_template_file": None,
            },
        )
        kwinput = {
            "question": "Question",
            "context": "Context",
            "answer": "Answer",
        }
        tinput = app.training_prompt(**kwinput)

        expected = "Answer"
        result = app.extract_answer(tinput, **kwinput)
        assert result == expected, "Extraction result is not as expected."

    def test_extract_answer_simple2(self):
        """Test with different keywords."""
        app = App.from_json(
            "tests/files/config.json",
            extra_params={
                "prompt_template": "Goal\n\n### Question:\n{{question}}\n\n### Additional Info:\n{{context}}\n\n### Answer:\n{{answer}}\n\n",
                "prompt_template_file": None,
            },
        )
        kwinput = {
            "user": "Question",
            "context": "Context",
            "assistant": "Answer",
        }
        tinput = app.training_prompt(**kwinput)

        expected = "Answer"
        result = app.extract_answer(tinput, **kwinput)
        assert result == expected, "Extraction result is not as expected."

    def test_extract_answer_simple3(self):
        """Test with bos tag."""
        app = App.from_json(
            "tests/files/config.json",
            extra_params={
                "prompt_template": "Goal\n\n### Question:\n{{question}}\n\n### Additional Info:\n{{context}}\n\n### Answer:\n{{answer}}\n\n",
                "prompt_template_file": None,
            },
        )
        kwinput = {
            "user": "Question",
            "context": "Context",
            "assistant": "Answer",
        }
        tinput = app.training_prompt(**kwinput)
        tinput = "<s> " + tinput + "</s>"

        expected = "Answer"
        result = app.extract_answer(tinput, **kwinput)
        assert result == expected, "Extraction result is not as expected."


if __name__ == "__main__":
    unittest.main()
