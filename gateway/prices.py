"""Per-model token prices, shipped with the function (copied from experiments/prices.json).

Prices are not discoverable at request time, so they are data, versioned with
the code. A model missing from the table is metered in tokens with no dollar
figure rather than guessed at.
"""
import json
import pathlib

_TABLE = None


def _table() -> dict:
    global _TABLE
    if _TABLE is None:
        here = pathlib.Path(__file__)
        # packaged copy in the Lambda; the repo's shared table when run from source
        for p in (here.with_name("prices.json"), here.parents[1] / "experiments" / "prices.json"):
            if p.exists():
                _TABLE = json.loads(p.read_text())["models"]
                break
        else:
            _TABLE = {}
    return _TABLE


def bare(model: str) -> str:
    """'mantle/mistral.x' -> 'mistral.x': the target prefix is routing, not identity."""
    return model.split("/", 1)[1] if "/" in model else model


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    p = _table().get(bare(model))
    if not p or p.get("input_per_1k") is None or p.get("output_per_1k") is None:
        return None
    return input_tokens / 1000 * p["input_per_1k"] + output_tokens / 1000 * p["output_per_1k"]


def output_price_per_1k(model: str) -> float | None:
    p = _table().get(bare(model))
    return p.get("output_per_1k") if p else None


def input_price_per_1k(model: str) -> float | None:
    p = _table().get(bare(model))
    return p.get("input_per_1k") if p else None
