all
rule 'MD007', :indent => 4 # Unordered list indentation
rule 'MD009', :br_spaces => 2 # Allow paragraph breaking spaces.
exclude_rule 'MD013' # Line length (Does not work with long links)
exclude_rule 'MD018' # No space after hash on atx style header (Did not work with tags)
exclude_rule 'MD022' # Headers should be surrounded by blank lines (Disabled for false positives)
exclude_rule 'MD023' # Headers must start at the beginning of the line (Catches Table of Contents)
exclude_rule 'MD024' # Multiple headers with the same content (Catches CHANGELOG.md)
exclude_rule 'MD025' # Multiple top level headers in the same document (Unused for Obsidian)
exclude_rule 'MD026' # Trailing punctuation in header (Did not work with tags)
exclude_rule 'MD029' # Ordered list item prefix (I like to number my ordered lists)
exclude_rule 'MD031' # Fenced code blocks should be surrounded by blank lines (Don't like rule)
exclude_rule 'MD032' # Lists should be surrounded by blank lines (Don't like rule)
exclude_rule 'MD036' # Emphasis used instead of a header (Catches Table of Contents)
exclude_rule 'MD041' # First line in file should be a top level header (Obsidian Properties can be first)
exclude_rule 'MD055' # Table row doesn't begin/end with pipes (Does not work with | in links)
exclude_rule 'MD057' # Table has missing or invalid header separation (second row)
