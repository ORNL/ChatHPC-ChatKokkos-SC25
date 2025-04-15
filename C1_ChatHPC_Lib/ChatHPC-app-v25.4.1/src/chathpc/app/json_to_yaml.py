import argparse
import sys

import yaml
from loguru import logger

from chathpc.app.utils.common_utils import load_json_yaml_arg


class BlockStr(str):
    """Subclass string to mark strings for block style in YAML."""

    __slots__ = ()


def block_str_representer(dumper, data):
    """Custom representer to use '|' style for multiline strings."""
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


yaml.add_representer(BlockStr, block_str_representer, Dumper=yaml.SafeDumper)


def json_to_yaml(json_or_str):
    # Step 1: Parse the JSON string
    json_obj = load_json_yaml_arg(json_or_str, False)

    # Recursively wrap strings that have real newlines (\n) for block style
    def wrap_multiline(obj):
        if isinstance(obj, str):
            return BlockStr(obj) if "\n" in obj else obj
        if isinstance(obj, list):
            return [wrap_multiline(i) for i in obj]
        if isinstance(obj, dict):
            return {k: wrap_multiline(v) for k, v in obj.items()}
        return obj

    wrapped_json_obj = wrap_multiline(json_obj)

    # Step 2: Convert to YAML
    return yaml.safe_dump(wrapped_json_obj, sort_keys=False)


def init_parser(parser):
    parser.add_argument("--debug", action="store_true", help="Open debug port (5678).")
    parser.add_argument("--log_level", type=str, help="Log level.")
    parser.add_argument("json", type=str, nargs="?", help="Json string or path to json file.")


def cli(raw_args=None):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Convert Json files to Yaml for ease of editing.""")
    init_parser(parser)

    # Parse and load AppConfig settings from the command line into the settings source.
    args = parser.parse_args(raw_args)

    # Setup debugging.
    if args.debug:
        if args.log_level is None:
            args.log_level = "DEBUG"

        import debugpy  # noqa: T100

        debugpy.listen(5678)  # noqa: T100
        print("Attach debugger to continue.", file=sys.stderr)
        debugpy.wait_for_client()  # noqa: T100

    # Setup log level
    if args.log_level:
        logger.remove()
        logger.add(sys.stderr, level=args.log_level.upper())
    else:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    if args.json is not None:
        print(json_to_yaml(args.json))
    else:
        json_str = sys.stdin.read()
        print(json_to_yaml(json_str))


if __name__ == "__main__":
    cli()
