import argparse
import sys

from loguru import logger

from chathpc.app.utils.common_utils import load_json_yaml_arg


def json_yaml_to_markdown(json_or_str, add_rating_template=False):
    return_str = []

    json_str = load_json_yaml_arg(json_or_str, False)

    return_str.append("# ChatHPC JSON to Markdown\n\n")

    for index, row in enumerate(json_str):
        question_keywords = ["question", "user", "prompt"]
        response_keywords = ["response", "assistant"]

        return_str.append(f"## Index {index}\n\n")
        # Context
        if "context" in row:
            return_str.append("### Context\n\n")
            return_str.append(row["context"])
            return_str.append("\n\n")
        # Question
        if any(key in row for key in question_keywords):
            return_str.append("### Question\n\n")
            for item in question_keywords:
                if item in row:
                    return_str.append(row[item])
                    return_str.append("\n\n")
                    break
        # Response
        if any(key in row for key in response_keywords):
            return_str.append("### Response\n\n")
            for item in response_keywords:
                if item in row:
                    return_str.append(row[item])
                    return_str.append("\n\n")
                    break
        # Answer
        if "answer" in row:
            return_str.append("### Answer\n\n")
            return_str.append(row["answer"])
            return_str.append("\n\n")
        # Rating
        if add_rating_template:
            return_str.append("### Rating\n\n")
            return_str.append("### Rating Notes\n\n")

    return "".join(return_str)


def init_parser(parser):
    parser.add_argument("--debug", action="store_true", help="Open debug port (5678).")
    parser.add_argument("--log_level", type=str, help="Log level.")
    parser.add_argument("--add_rating_template", action="store_true", help="Add rating template to markdown.")
    parser.add_argument("json", type=str, nargs="?", help="Json string or path to json file.")


def cli(raw_args=None):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Convert Json files to Markdown for ease of reading.""")
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
        print(json_yaml_to_markdown(args.json, add_rating_template=args.add_rating_template))
    else:
        json_str = sys.stdin.read()
        print(json_yaml_to_markdown(json_str, add_rating_template=args.add_rating_template))


if __name__ == "__main__":
    cli()
