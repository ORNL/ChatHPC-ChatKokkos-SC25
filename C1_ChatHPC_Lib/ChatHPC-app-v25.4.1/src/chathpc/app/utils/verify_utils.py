def ignore_minor(string: str):
    s = string.strip()
    line = s.splitlines()
    line = [x.strip() for x in line]
    return "\n".join(line)
