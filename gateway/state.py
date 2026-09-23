"""Shared state for the pipeline: spend ledger and request/response correlation.

Two facts about AgentCore Gateway interceptors force this module to exist:

  * The RESPONSE interceptor does not receive the original request -- its input
    carries `gatewayRequest: null`. Anything the response phase needs to know
    (who called, which model was chosen) must be remembered by the request
    phase, keyed on the gateway's REQUEST_ID, which both invocations share.

  * The gateway may retry an interceptor after a failure or timeout. Settling a
    charge twice would double-bill, so settlement is idempotent: a conditional
    marker on the request record and the spend increment commit in a single
    DynamoDB transaction, and a retry finds the marker and changes nothing.

One on-demand table holds every kind of item, told apart by key prefix:
`spend#`, `req#`, `cache#` and `vec#`.

The `vec#` items carry an embedding and are the only ones the table's vector
index sees: DynamoDB indexes an item only if it has the vector attribute, so
the spend ledger and the response cache sit in the same table without being
searched. That is the reason the semantic cache needs no second datastore --
recall, the cached answer and the spend ledger share one table, one IAM policy
and one TTL sweeper.
"""
import datetime as dt
import json
import time
from decimal import Decimal

TTL_REQUEST_S = 3600          # a correlation record only has to outlive one call
TTL_SPEND_DAYS = 7


def _day_key(tenant: str) -> str:
    return f"spend#{tenant}#{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}"


class DynamoStore:
    def __init__(self, table: str):
        import boto3
        self.table = table
        self.ddb = boto3.client("dynamodb")

    def spend_today(self, tenant: str) -> float:
        r = self.ddb.get_item(TableName=self.table, Key={"pk": {"S": _day_key(tenant)}},
                              ProjectionExpression="spend_usd")
        return float(r.get("Item", {}).get("spend_usd", {}).get("N", 0))

    def remember(self, request_id: str, data: dict) -> None:
        self.ddb.put_item(TableName=self.table, Item={
            "pk": {"S": f"req#{request_id}"},
            **{k: {"S": str(v)} for k, v in data.items()},
            "expires_at": {"N": str(int(time.time()) + TTL_REQUEST_S)},
        })

    def recall(self, request_id: str) -> dict:
        r = self.ddb.get_item(TableName=self.table, Key={"pk": {"S": f"req#{request_id}"}})
        return {k: v["S"] for k, v in r.get("Item", {}).items() if "S" in v and k != "pk"}

    def get_cached(self, key: str):
        r = self.ddb.get_item(TableName=self.table, Key={"pk": {"S": f"cache#{key}"}},
                              ProjectionExpression="payload")
        item = r.get("Item")
        return json.loads(item["payload"]["S"]) if item else None

    def put_cached(self, key: str, body: dict, ttl_s: int) -> None:
        self.ddb.put_item(TableName=self.table, Item={
            "pk": {"S": f"cache#{key}"},
            "payload": {"S": json.dumps(body)},
            "expires_at": {"N": str(int(time.time()) + ttl_s)},
        })

    # ------------------------------------------------------------ rate windows
    #
    # One item per tenant per window, incremented with ADD so the counter is
    # correct under concurrency without a read-modify-write. The returned
    # attributes are the values *after* this call's increment, which is what
    # makes the limit check race-free: whoever pushes the counter past the
    # limit is the one refused.

    def bump_window(self, tenant: str, window: int, window_s: int, requests: int = 1,
                    tokens: int = 0, smooth: bool = False) -> dict:
        r = self.ddb.update_item(
            TableName=self.table,
            Key={"pk": {"S": f"rate#{tenant}#{window}"}},
            UpdateExpression="ADD reqs :r, toks :t SET expires_at = :e",
            ExpressionAttributeValues={
                ":r": {"N": str(requests)},
                ":t": {"N": str(tokens)},
                # Two windows of slack so a smoothed check can still read the
                # previous window after this one opens.
                ":e": {"N": str(int(time.time()) + window_s * 3)},
            },
            ReturnValues="UPDATED_NEW")
        attrs = r.get("Attributes", {})
        out = {"requests": float(attrs.get("reqs", {}).get("N", 0)),
               "tokens": float(attrs.get("toks", {}).get("N", 0))}
        if smooth:
            # Sliding-window approximation: carry the fraction of the previous
            # window still inside the trailing `window_s` seconds. Costs one
            # extra read per call and removes the fixed window's boundary burst.
            prev = self.ddb.get_item(
                TableName=self.table,
                Key={"pk": {"S": f"rate#{tenant}#{window - 1}"}},
                ProjectionExpression="reqs, toks").get("Item", {})
            elapsed = time.time() - window * window_s
            carry = max(0.0, 1.0 - elapsed / window_s)
            out["requests"] += carry * float(prev.get("reqs", {}).get("N", 0))
            out["tokens"] += carry * float(prev.get("toks", {}).get("N", 0))
        return out

    # ---------------------------------------------------------------- vectors
    #
    # DynamoDB stores an embedding as a list of numbers and searches it with
    # SearchVectors against a vector index declared on the table. The index is
    # partitioned by `tenant` (its HASH search-schema element), so a search
    # never crosses a tenant boundary and never scans the whole corpus.

    def put_vector(self, key: str, tenant: str, vec: list[float], prompt: str,
                   ttl_s: int, index_key: str | None = None) -> None:
        self.ddb.put_item(TableName=self.table, Item={
            "pk": {"S": f"vec#{tenant}#{index_key or key}"},
            "tenant": {"S": tenant},
            # float32 in, float32 out: eight decimals is past the point where
            # more digits survive the round trip.
            "embedding": {"L": [{"N": f"{x:.8f}"} for x in vec]},
            "cache_key": {"S": key},
            "prompt": {"S": prompt[:2000]},
            "expires_at": {"N": str(int(time.time()) + ttl_s)},
        })

    def nearest(self, index: str, tenant: str, vec: list[float], top_k: int = 1) -> list[dict]:
        """Nearest stored prompts, closest first, as {similarity, cache_key, prompt}."""
        r = self.ddb.search_vectors(
            TableName=self.table, IndexName=index, TopK=top_k,
            SearchVector=[{"N": f"{x:.8f}"} for x in vec],
            SearchConditionExpression="tenant = :t",
            ExpressionAttributeValues={":t": {"S": tenant}})
        out = []
        for m in r.get("SearchResults", []):
            item = m.get("Item", {})
            out.append({
                # COSINE scores are distances: 0 is identical, 2 is opposite.
                "similarity": 1.0 - float(m.get("Score", 2.0)),
                "cache_key": item.get("cache_key", {}).get("S", ""),
                "prompt": item.get("prompt", {}).get("S", ""),
            })
        return out

    def settle(self, request_id: str, tenant: str, cost: float) -> bool:
        """Charge once per request. Returns False if this request was already settled."""
        expires = int((dt.datetime.now(dt.timezone.utc)
                       + dt.timedelta(days=TTL_SPEND_DAYS)).timestamp())
        try:
            self.ddb.transact_write_items(TransactItems=[
                {"Update": {
                    "TableName": self.table,
                    "Key": {"pk": {"S": f"req#{request_id}"}},
                    "UpdateExpression": "SET settled = :t",
                    "ConditionExpression": "attribute_not_exists(settled)",
                    "ExpressionAttributeValues": {":t": {"BOOL": True}},
                }},
                {"Update": {
                    "TableName": self.table,
                    "Key": {"pk": {"S": _day_key(tenant)}},
                    "UpdateExpression": "ADD spend_usd :c SET expires_at = :e",
                    "ExpressionAttributeValues": {
                        ":c": {"N": str(Decimal(str(round(cost, 8))))},
                        ":e": {"N": str(expires)},
                    },
                }},
            ])
            return True
        except self.ddb.exceptions.TransactionCanceledException as exc:
            reasons = exc.response.get("CancellationReasons", [])
            if reasons and reasons[0].get("Code") == "ConditionalCheckFailed":
                return False          # a retry of a call we already charged
            raise


_store = None


def store(table: str | None = None):
    """The process-wide store. Tests replace it with an in-memory fake."""
    global _store
    if _store is None:
        if not table:
            raise RuntimeError("no state table configured")
        _store = DynamoStore(table)
    return _store


def set_store(s) -> None:
    global _store
    _store = s
