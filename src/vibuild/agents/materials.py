"""Read several exact material selectors with one selected response; never execute code."""

import argparse
import ast
import json
from pathlib import Path
import re


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def shape(value):
    if isinstance(value, dict):
        return {"type": "object", "count": len(value), "keys": list(value)}
    if isinstance(value, list):
        return {
            "type": "array",
            "count": len(value),
            "item": shape(value[0]) if value else None,
        }
    if isinstance(value, str):
        return {"type": "string", "chars": len(value)}
    return {"type": type(value).__name__}


def select(spec):
    filename, _, selector = spec.partition("#")
    path = Path(filename)
    text = path.read_text()
    if path.suffix == ".json":
        value = json.loads(text)
        if not selector:
            # An unqualified JSON read is an index, even when the file is small.
            return (
                {k: shape(v) for k, v in value.items()}
                if isinstance(value, dict)
                else shape(value)
            ), True
        if not selector.startswith("/"):
            raise ValueError("JSON selectors use /field/index (JSON Pointer)")
        for key in selector[1:].split("/"):
            key = key.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
        return value, False
    lines = text.splitlines()
    span = re.fullmatch(r"L(\d+)-L?(\d+)", selector)
    if span:
        start, end = map(int, span.groups())
        if not 1 <= start <= end <= len(lines):
            raise ValueError(f"Line range must be within 1-{len(lines)}")
        return "\n".join(f"{i + 1}: {lines[i]}" for i in range(start - 1, end)), False
    if path.suffix == ".py":
        nodes = [
            n
            for n in ast.parse(text).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if not selector:
            return {
                "lines": len(lines),
                "definitions": [
                    {"name": n.name, "lines": [n.lineno, n.end_lineno]} for n in nodes
                ],
            }, True
        node = next((n for n in nodes if n.name == selector), None)
        if node is None:
            raise ValueError("Unknown definition; read the file index first")
        return "\n".join(lines[node.lineno - 1 : node.end_lineno]), False
    if selector:
        raise ValueError("Text selectors use Lstart-Lend")
    return text, False


def read_materials(specs):
    """Unqualified reads return structure; explicit selectors return complete values."""
    rows = []
    for spec in specs:
        try:
            value, summary = select(spec)
            rows.append({"source": spec, "index" if summary else "value": value})
        except (
            OSError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            SyntaxError,
        ) as exc:
            rows.append({"source": spec, "error": str(exc)})
    return encode(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "selectors",
        nargs="+",
        help="file.json#/field/0, file.py#function, file#L10-L30; no selector gives an index",
    )
    a = p.parse_args(argv)
    try:
        print(read_materials(a.selectors))
    except ValueError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
