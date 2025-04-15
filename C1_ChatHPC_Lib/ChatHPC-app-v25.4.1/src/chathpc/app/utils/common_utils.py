from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import yaml
from option import Err, Ok, Result


def get_valid_path_from_string(path_str: str) -> Result[Path, Exception]:
    """Check if a string represents a valid file path.

    Args:
        path_str (str): A string representing a file path.

    Returns:
        Result[Path, Exception]: An Ok result containing the Path object if the path
            exists and is a file, or an Err result containing the exception if the
            path is invalid or doesn't exist.

    Examples:
        >>> get_valid_path_from_string("/path/to/existing/file.txt").is_ok
        True
        >>> get_valid_path_from_string("/nonexistent/path.txt").is_ok
        False
    """
    try:
        path = Path(path_str)
        if path.is_file():
            return Ok(path)
        return Err(FileNotFoundError(path_str))
    except Exception as e:  # noqa: BLE001
        return Err(e)


def get_valid_json_from_string(json_str: str) -> Result[dict, Exception]:
    """Check if a string represents valid JSON and parse it.

    Args:
        json_str (str): A string containing JSON data.

    Returns:
        Result[dict, Exception]: An Ok result containing the parsed JSON as a dictionary
            if the string is valid JSON, or an Err result containing the exception if
            the string is invalid JSON.

    Examples:
        >>> get_valid_json_from_string('{"key": "value"}').is_ok
        True
        >>> get_valid_json_from_string("invalid json").is_ok
        False
    """
    try:
        values = json.loads(json_str)
        return Ok(values)
    except Exception as e:  # noqa: BLE001
        return Err(e)


def get_valid_yaml_from_string(yaml_str: str) -> Result[dict, Exception]:
    """Check if a string represents valid YAML and parse it.

    Args:
        yaml_str (str): A string containing YAML data.

    Returns:
        Result[dict, Exception]: An Ok result containing the parsed YAML as a dictionary
            if the string is valid YAML, or an Err result containing the exception if
            the string is invalid YAML.

    Examples:
        >>> get_valid_yaml_from_string("key: value").is_ok
        True
        >>> get_valid_yaml_from_string("invalid: yaml: structure:").is_ok
        False
    """
    try:
        values = yaml.safe_load(yaml_str)
        return Ok(values)
    except Exception as e:  # noqa: BLE001
        return Err(e)


def load_json_yaml_arg(str_or_fn: str, add_filename: bool = True):
    """Load and parse JSON data from either a string or file.

    Args:
        str_or_fn (str): Either a JSON string starting with '{' or a path to a JSON file.
            If None, returns an empty dict.

    Returns:
        dict: Parsed JSON data as a dictionary. Returns empty dict if input is None.

    Examples:
        >>> load_json_yaml_arg('{"key": "value"}')
        {'key': 'value'}
        >>> load_json_yaml_arg("path/to/file.json")
        {'contents': 'from file'}
        >>> load_json_yaml_arg(None)
        {}
    """
    if str_or_fn is None:
        return {}

    if isinstance(str_or_fn, dict):
        return str_or_fn

    if isinstance(str_or_fn, list):
        return str_or_fn

    path = get_valid_path_from_string(str_or_fn)
    if path.is_ok:
        path = path.unwrap()

        if path.suffix == ".json":
            with open(path) as f:
                params = json.loads(f.read())
                if add_filename:
                    params["filename"] = str_or_fn
                f.close()
            return params

        if path.suffix in (".yaml", ".yml"):
            with open(path) as f:
                params = yaml.safe_load(f)
                if add_filename:
                    params["filename"] = str_or_fn
                f.close()
            return params

        msg = f"Unsupported file type: {str_or_fn}"
        raise ValueError(msg)

    json_values = get_valid_json_from_string(str_or_fn)
    if json_values.is_ok:
        return json_values.unwrap()

    yaml_values = get_valid_yaml_from_string(str_or_fn)
    if yaml_values.is_ok:
        return yaml_values.unwrap()

    msg = f"Invalid JSON or YAML string or missing file: {str_or_fn}"
    raise ValueError(msg)


def evaluate_fstring(fstring, **kwargs):
    """Evaluate a string as an f-string with provided keyword arguments.

    Args:
        fstring (str): The string to be evaluated as an f-string. Can contain
            Python expressions inside curly braces {}.
        **kwargs: Keyword arguments that will be used to format the f-string.

    Returns:
        str: The evaluated f-string with all expressions replaced with their values.

    Examples:
        >>> evaluate_fstring("Hello {name}!", name="World")
        'Hello World!'
        >>> evaluate_fstring("The sum is {x + y}", x=1, y=2)
        'The sum is 3'
    """
    return eval(f"f'''{fstring}'''", {}, kwargs)  # noqa: S307


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
