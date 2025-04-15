from __future__ import annotations

import re

ALIASES = {
    "system": "system",
    "context": "context",
    "prompt": "prompt",
    "question": "prompt",
    "user": "prompt",
    "response": "response",
    "answer": "response",
    "assistant": "response",
}


def keyword_alias(key: str) -> str:
    """Return the standardized keyword for a given alias.

    Args:
        key (str): The input alias or keyword to standardize

    Returns:
        str: The standardized keyword if the input is an alias,
             otherwise returns the input unchanged

    Example:
        >>> keyword_alias("assistant")
        'response'
        >>> keyword_alias("unknown")
        'unknown'
    """
    return ALIASES.get(key.lower(), key)


def map_keywords(kwargs: dict[str, str]) -> dict[str, str]:
    """Map input keyword aliases to their standardized forms.

    Args:
        kwargs (dict[str,str]): Dictionary with keyword aliases as keys

    Returns:
        dict[str, str]: Dictionary with standardized keywords as keys

    Example:
        >>> map_keywords({"assistant": "Hello", "user": "Hi"})
        {'response': 'Hello', 'prompt': 'Hi'}
    """
    return {keyword_alias(k): v for k, v in kwargs.items()}


def get_aliases_for_keyword(key):
    """Return a list of all aliases that map to the given keyword.

    Args:
        key (str): The keyword to find aliases for

    Returns:
        list[str]: List of aliases that map to the given keyword

    Example:
        >>> get_aliases_for_keyword("response")
        ['response', 'answer', 'assistant']
    """
    return [k for k, v in ALIASES.items() if v == key]


def normalize_template(template: str) -> str:
    """Normalize template by converting all alias keywords to their standardized form.

    Args:
        template (str): Input template string containing alias keywords in {{alias}} format

    Returns:
        str: Template string with all alias keywords converted to their standardized form

    Example:
        >>> normalize_template("{{assistant}} responds to {{user}}")
        '{{ response }} responds to {{ prompt }}'
    """
    # Expand raw newlines.
    template = template.replace("\\n", "\n")

    # Update keywords.
    for alias, keyword in ALIASES.items():
        pattern = r"{{\s*" + alias + r"\s*}}"
        template = re.sub(pattern=pattern, repl=f"{{{{ {keyword} }}}}", string=template)

    return template


def split_on_response(template: str) -> tuple[str, str]:
    """Split a template string into parts before and after the response keyword.

    Args:
        template (str): Template string containing a response keyword in {{response}} format

    Raises:
        ValueError: If template cannot be split into exactly 3 parts (before, keyword, after)

    Returns:
        tuple[str, str]: A tuple containing (pre_response_text, post_response_text)

    Example:
        >>> split_on_response("User: {{prompt}} Bot: {{response}} End")
        ('User: {{prompt}} Bot: ', ' End')
    """
    pattern = r"{{\s*(response)\s*}}"

    parts = list(re.split(pattern, template))
    if len(parts) != 3:  # noqa: PLR2004
        msg = f"expected 3 parts, got {len(parts)}"
        raise ValueError(msg)
    return parts[0], parts[-1]


def convert_go_to_jinja(go_template: str) -> str:
    """Convert Go template syntax to Jinja2 syntax."""

    # Convert variables: {{ .Var }} -> {{ Var }}
    go_template = re.sub(r"{{\s*\.(\w+)\s*}}", r"{{ \1 }}", go_template)

    # Convert if statements: {{ if .Var }} -> {% if Var %}
    go_template = re.sub(r"{{\s*if\s*\.(\w+)\s*}}", r"{% if \1 %}", go_template)

    # Convert else if: {{ else if .Var }} -> {% elif Var %}
    go_template = re.sub(r"{{\s*else if\s*\.(\w+)\s*}}", r"{% elif \1 %}", go_template)

    # Convert else: {{ else }} -> {% else %}
    go_template = go_template.replace("{{ else }}", "{% else %}")

    # Convert end statements: {{ end }} -> {% endif %} (for conditionals)
    go_template = re.sub(r"{{\s*end\s*}}", r"{% endif %}", go_template)

    # Convert range loops: {{ range .List }} -> {% for item in List %}
    go_template = re.sub(r"{{\s*range\s*\.(\w+)\s*}}", r"{% for item in \1 %}", go_template)

    # Convert end loops: {{ end }} -> {% endfor %}
    go_template = re.sub(r"{{\s*end\s*}}", r"{% endfor %}", go_template)

    # Convert comments: {{/* comment */}} -> {# comment #}
    go_template = re.sub(r"{{/\*\s*(.*?)\s*\*/}}", r"{# \1 #}", go_template)

    return go_template  # noqa: RET504


def convert_jinja_to_go(jinja_template: str) -> str:
    """Convert Jinja2 template syntax to Go template syntax."""

    # Convert variables: {{ Var }} -> {{ .Var }}
    jinja_template = re.sub(r"{{\s*(\w+)\s*}}", r"{{ .\1 }}", jinja_template)

    # Convert if statements: {% if Var %} -> {{ if .Var }}
    jinja_template = re.sub(r"{%\s*if\s*(\w+)\s*%}", r"{{ if .\1 }}", jinja_template)

    # Convert elif statements: {% elif Var %} -> {{ else if .Var }}
    jinja_template = re.sub(r"{%\s*elif\s*(\w+)\s*%}", r"{{ else if .\1 }}", jinja_template)

    # Convert else statements: {% else %} -> {{ else }}
    jinja_template = jinja_template.replace("{% else %}", "{{ else }}")

    # Convert endif statements: {% endif %} -> {{ end }}
    jinja_template = re.sub(r"{%\s*endif\s*%}", r"{{ end }}", jinja_template)

    # Convert for loops: {% for item in List %} -> {{ range .List }}
    jinja_template = re.sub(r"{%\s*for\s*\w+\s*in\s*(\w+)\s*%}", r"{{ range .\1 }}", jinja_template)

    # Convert endfor statements: {% endfor %} -> {{ end }}
    jinja_template = re.sub(r"{%\s*endfor\s*%}", r"{{ end }}", jinja_template)

    # Convert comments: {# comment #} -> {{/* comment */}}
    jinja_template = re.sub(r"{#\s*(.*?)\s*#}", r"{{/* \1 */}}", jinja_template)

    return jinja_template  # noqa: RET504
