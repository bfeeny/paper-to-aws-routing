#!/usr/bin/env python3
"""Refresh experiments/prices.json from the Marketplace offer API.

Bedrock model offers carry the full rate card, so prices can be read from the
API instead of transcribed from a pricing page: per account, dated, and
machine-readable. Models with no offer (mantle-only IDs) keep null prices —
the ledger still counts tokens, it just won't claim dollars it can't source.

    python3 runner/fetch_prices.py [--region-code USE1]
"""

import argparse
import datetime as dt
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRICES = ROOT / "experiments" / "prices.json"


def aws_json(*args, profile: str, region: str):
    r = subprocess.run(["aws", *args, "--profile", profile, "--region", region,
                        "--output", "json"], capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip()[:160]
    return (json.loads(r.stdout) if r.stdout.strip() else {}), None


def price_list_rate(model: str, region_code: str, profile: str, region: str):
    """Look a model up in the AWS Price List API by its exact usage type.

    The usage type embeds the mantle model id verbatim, e.g.
    `USE1-qwen.qwen3-32b-mantle-input-tokens-standard`, so no display-name
    mapping (and no guessing) is needed. Prices here are already per 1K tokens.
    """
    out = {}
    for kind in ("input", "output"):
        usagetype = f"{region_code}-{model}-mantle-{kind}-tokens-standard"
        d, err = aws_json("pricing", "get-products", "--service-code", "AmazonBedrock",
                          "--filters", f"Type=TERM_MATCH,Field=usagetype,Value={usagetype}",
                          profile=profile, region="us-east-1")
        items = (d or {}).get("PriceList", [])
        if not items:
            return None, None, f"no price-list entry for {usagetype}"
        product = json.loads(items[0])
        term = list(product["terms"]["OnDemand"].values())[0]
        dim = list(term["priceDimensions"].values())[0]
        if "1K tokens" not in dim["unit"]:
            return None, None, f"unexpected unit {dim['unit']!r}"
        out[kind] = float(dim["pricePerUnit"]["USD"])
    return out["input"], out["output"], "price-list:mantle-standard"


def offer_rate(model: str, region_code: str, profile: str, region: str):
    """Fallback: the Marketplace offer rate card (per MILLION tokens)."""
    offers, err = aws_json("bedrock", "list-foundation-model-agreement-offers",
                           "--model-id", model, profile=profile, region=region)
    if not offers or not offers.get("offers"):
        return None, None, "no offer"
    card = offers["offers"][0].get("termDetails", {}) \
        .get("usageBasedPricingTerm", {}).get("rateCard", [])
    rates = {r["dimension"]: r["price"] for r in card}
    kin, kout = f"{region_code}_input_tokens_standard", f"{region_code}_output_tokens_standard"
    if kin not in rates or kout not in rates:
        return None, None, f"no {region_code} standard dimensions in offer"
    return float(rates[kin]) / 1000.0, float(rates[kout]) / 1000.0, "offer:rate-card"


def rate_for(model: str, region_code: str, profile: str, region: str):
    """Price List first (covers every servable model), offer card as fallback."""
    pin, pout, src = price_list_rate(model, region_code, profile, region)
    if pin is not None:
        return pin, pout, src
    first_why = src
    pin, pout, src = offer_rate(model, region_code, profile, region)
    if pin is not None:
        return pin, pout, src
    return None, None, f"{first_why}; {src}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="personal")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--region-code", default="USE1",
                    help="rate-card Region prefix; USE1 = us-east-1")
    args = ap.parse_args()

    prices = json.loads(PRICES.read_text())
    models = sorted(prices.get("models", {}).keys())

    # include any model named by an experiment definition
    for exp in (ROOT / "experiments").glob("*.json"):
        if exp.name == "prices.json":
            continue
        cfg = json.loads(exp.read_text())
        for m in (cfg.get("pair", {}).get("strong"), cfg.get("pair", {}).get("weak"),
                  cfg.get("judge", {}).get("model")):
            if m and m not in models:
                models.append(m)

    resolved, unpriced = 0, []
    for m in sorted(models):
        pin, pout, why = rate_for(m, args.region_code, args.profile, args.region)
        entry = prices["models"].get(m, {"quota_output_weight": 1.0})
        entry["input_per_1k"], entry["output_per_1k"] = pin, pout
        entry.setdefault("quota_output_weight", 1.0)
        if pin is None:
            entry["price_note"] = why
            unpriced.append((m, why))
        else:
            entry.pop("price_note", None)
            entry["price_source"] = why
            resolved += 1
        prices["models"][m] = entry
        print(f"  {m:44} " + (f"in={pin:.6f} out={pout:.6f} /1k" if pin else f"unpriced ({why})"))

    prices["_priced_on"] = dt.date.today().isoformat()
    prices["_price_region"] = args.region_code
    PRICES.write_text(json.dumps(prices, indent=2) + "\n")
    print(f"\npriced {resolved}/{len(models)}; wrote {PRICES.relative_to(ROOT)}")
    if unpriced:
        print("unpriced models keep token-only accounting:")
        for m, why in unpriced:
            print(f"  {m:44} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
