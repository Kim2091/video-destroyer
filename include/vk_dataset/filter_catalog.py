"""
The filter templates the Filter Template node offers: Vapourkit's .vkfilter
files in filters/, each a short VapourSynth script that reads `clip` and
assigns the result back to it.

A .vkfilter is TOML with a name, a category, a description and the code. A
template may declare [variables.<name>] with a default; its code then holds
{{name}} where the value goes, and the catalog writes the default in, so
every template's code here is plain Python.
"""

import json
import os
import re
import tomllib
from dataclasses import dataclass

CATALOG_DIR = os.path.join(os.path.dirname(__file__), "filters")

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


@dataclass(frozen=True)
class FilterTemplate:
    name: str
    categories: tuple
    description: str
    code: str


def _literal(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))


def render(code, variables):
    """The code with each declared {{variable}} replaced by its default as a Python literal."""
    def replace(match):
        declared = variables.get(match.group(1))
        return _literal(declared["default"]) if declared and "default" in declared else match.group(0)
    return _PLACEHOLDER.sub(replace, code.strip())


def read_template(path):
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    categories = data.get("category", [])
    code = render(data["code"], data.get("variables", {}))
    if _PLACEHOLDER.search(code):
        raise ValueError(f"{os.path.basename(path)} has a placeholder with no default")
    return FilterTemplate(data["name"], tuple([categories] if isinstance(categories, str) else categories),
                          data.get("description", ""), code)


_catalog = None


def catalog():
    """Every template in filters/, by name, in name order."""
    global _catalog
    if _catalog is None:
        templates = [read_template(os.path.join(CATALOG_DIR, f))
                     for f in os.listdir(CATALOG_DIR) if f.endswith(".vkfilter")]
        _catalog = {t.name: t for t in sorted(templates, key=lambda t: t.name.lower())}
    return _catalog
