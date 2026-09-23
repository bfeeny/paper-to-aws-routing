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
`spend#`, `req#` and `cache#`.
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
