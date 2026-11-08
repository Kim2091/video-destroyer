"""
Value nodes: one number, text or bool per chunk, linked into the settings of
other nodes (links.py) so they can share a draw or derive one setting from
another.

A value node draws from its own generator like every node, so a shared pick
is the same for every node it feeds, and rerunning a job reproduces it. It
runs once per chunk when everything it feeds does, and once per video when
something it feeds runs per video (graph.py).

Each declares `describe`, what its value can be, so a link the receiving
setting can never take is refused when the graph loads.
"""

import math

from ..links import LinkError, number, same
from ..params import sample
from ..registry import Param, node

OPERATIONS = ["add", "subtract", "multiply", "divide", "min", "max"]


def _range(info):
    """A described value as (low, high, whole), or None when it is not a known number."""
    if info is None:
        return None
    if info[0] == "range":
        return info[1], info[2], info[3]
    values = [number(v) for v in info[1]]
    if not values or any(v is None for v in values):
        return None
    return min(values), max(values), all(v == int(v) for v in values)


def _describe_random_number(get):
    low, high, whole = _range(get("low")), _range(get("high")), get("whole")
    if low is None or high is None or whole is None or whole[0] != "set" or len(whole[1]) != 1:
        return None
    lo, hi = min(low[0], high[0]), max(low[1], high[1])
    return ("range", math.ceil(lo), math.floor(hi), True) if whole[1][0] is True else ("range", lo, hi, False)


@node("random_number", category="Values", label="Random Number",
      outputs={"value": "value"},
      params=[
          Param("low", "float", 0.0, help="The smallest number it gives."),
          Param("high", "float", 1.0, help="The largest number it gives."),
          Param("whole", "bool", False, label="Whole numbers",
                help="Give only whole numbers, both ends included."),
      ],
      describe=_describe_random_number)
def random_number(ctx, *, low, high, whole):
    low, high = min(low, high), max(low, high)
    if whole:
        if math.ceil(low) > math.floor(high):
            raise LinkError(f"there is no whole number from {low:g} to {high:g}")
        value = ctx.rng.randint(math.ceil(low), math.floor(high))
    else:
        value = ctx.rng.uniform(low, high)
    return {"value": value}


def _describe_random_pick(get):
    # The options are a choice sampler, which links.describe reads as the options it can draw.
    options = get("options")
    return options if options is not None and options[0] == "set" else None


@node("random_pick", category="Values", label="Random Pick",
      outputs={"value": "value"},
      params=[
          Param("options", "options", {"choice": []}, gives=True,
                help="The values to pick from, each with an optional weight. Wired into a dropdown, they are "
                     "that dropdown's options."),
      ],
      describe=_describe_random_pick)
def random_pick(ctx, *, options):
    if not isinstance(options, dict) or not options.get("choice"):
        raise LinkError("Random Pick has no options to pick from")
    return {"value": sample(options, ctx.rng)}


def _bounds(op, a, b):
    """The (low, high) of `a op b` for a in [a0, a1] and b in [b0, b1], or None when unbounded."""
    (a0, a1), (b0, b1) = a, b
    if op == "add":
        return a0 + b0, a1 + b1
    if op == "subtract":
        return a0 - b1, a1 - b0
    if op == "min":
        return min(a0, b0), min(a1, b1)
    if op == "max":
        return max(a0, b0), max(a1, b1)
    if op == "divide":
        if b0 <= 0 <= b1:
            return None
        b = (1 / b1, 1 / b0)
    else:
        b = (b0, b1)
    products = [x * y for x in (a0, a1) for y in b]
    return min(products), max(products)


def _describe_math(get):
    a, b, op = _range(get("a")), _range(get("b")), get("operation")
    if a is None or b is None or op is None or op[0] != "set" or len(op[1]) != 1:
        return None
    bounds = _bounds(op[1][0], a[:2], b[:2])
    if bounds is None:
        return None
    whole = a[2] and b[2] and op[1][0] in ("add", "subtract", "multiply", "min", "max")
    return ("range", bounds[0], bounds[1], whole)


@node("math", category="Values", label="Math",
      outputs={"value": "value"},
      params=[
          Param("a", "float", 0.0, label="A", help="The first number. Link a value here to work from it."),
          Param("operation", "choice", "multiply", choices=OPERATIONS, help="What to do with A and B."),
          Param("b", "float", 1.0, label="B", help="The second number."),
      ],
      describe=_describe_math)
def math_node(ctx, *, a, operation, b):
    if operation == "add":
        value = a + b
    elif operation == "subtract":
        value = a - b
    elif operation == "multiply":
        value = a * b
    elif operation == "divide":
        if b == 0:
            raise LinkError("Math divides by 0")
        value = a / b
    elif operation == "min":
        value = min(a, b)
    else:
        value = max(a, b)
    return {"value": value}


def _describe_if_equals(get):
    then, otherwise = get("then"), get("otherwise")
    if then is None or otherwise is None or then[0] != "set" or otherwise[0] != "set":
        return None
    return ("set", (*then[1], *otherwise[1]))


@node("if_equals", category="Values", label="If Equals",
      outputs={"value": "value"},
      params=[
          Param("value", "value", "", help="The value to test. Link a value here."),
          Param("equals", "value", "", help="What it is compared with. Numbers compare as numbers."),
          Param("then", "value", "", gives=True, random=True, help="Given when the value equals it."),
          Param("otherwise", "value", "", label="Else", gives=True, random=True,
                help="Given when it does not."),
      ],
      describe=_describe_if_equals)
def if_equals(ctx, *, value, equals, then, otherwise):
    return {"value": then if same(value, equals) else otherwise}
