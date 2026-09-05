r"""Event store under the Glass Box data flow - local now, DynamoDB later with the same data model.

The JSONL file per call stays the raw, append-only stream. This store is the query layer on top:
keep the events, filter them, search transcripts across calls, list calls with their outcome.

DATA MODEL (DynamoDB single-table design, used identically by both backends)

    PK          CALL#<correlation id>
    SK          EV#<seq:06d>          one item per event
                META                  one item per call (start, end, caller, language, outcome, files)
    GSI1PK      KIND#<kind>           secondary index: all events of one kind across calls ...
    GSI1SK      <ts>                  ... ordered by wall-clock time
    GSI2PK      DAY#<yyyymmdd>        secondary index: calls (META items) of one day
    GSI2SK      <ts>
    text        searchable text pulled out of the payload (transcripts, tool args/results, SIP line)
    item        the full event / meta as stored (payload included)

BACKENDS
    sqlite      calls\glassbox.db - one file, no install. Table `items` with the key columns above,
                JSON1 for payload access, FTS5 virtual table for full-text search over `text`.
    dynamodb    boto3 against AWS or DynamoDB Local (endpoint_url). Same items, same keys;
                full-text search becomes a filtered Query/Scan on `text` (contains) - flagged in
                the result so the page can say so.

    store = make_store(settings)          # chosen by config.ini [store] backend
    events.subscribe(store.on_event)      # every event lands in the store (background writer)
    store.query_events(corr, kind="stt.done")
    store.search("credit card", kind="stt.done", limit=50)
    store.list_calls(limit=30)

Writes never block the agent: on_event only queues; a writer thread batches inserts.
Import old call files:  python -m agent.store --import
"""
from __future__ import annotations

import json
import queue
import sqlite3
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CALLS = PROJECT / "calls"

# payload fields that make an event findable by text
_TEXT_FIELDS = ("text", "result", "line", "utterance", "reason", "instruction", "tool", "detail",
                "record_id", "reference", "speculative", "final", "card", "action", "decision", "detected",
                "sip_from", "sip_to", "verdict", "from", "to")


def _searchable(ev: dict) -> str:
    p = ev.get("payload") or {}
    parts = [ev.get("kind", ""), ev.get("hop", "")]
    for k in _TEXT_FIELDS:
        v = p.get(k)
        if isinstance(v, (str, int, float)):
            parts.append(str(v))
    if isinstance(p.get("args"), dict):
        parts.append(json.dumps(p["args"], ensure_ascii=False))
    return " ".join(parts)[:2000]


def _keys(ev: dict) -> dict:
    corr = ev.get("corr") or "no-call"
    ts = float(ev.get("ts") or time.time())
    day = time.strftime("%Y%m%d", time.localtime(ts))
    return {"PK": f"CALL#{corr}", "SK": f"EV#{int(ev.get('seq') or 0):06d}",
            "GSI1PK": f"KIND#{ev.get('kind')}", "GSI1SK": ts, "GSI2PK": f"DAY#{day}", "GSI2SK": ts}


class EventStore:
    """Interface. Backends implement _put_items(list[dict]) and the read methods."""
    name = "base"

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._meta: dict = {}                      # corr -> META item under construction
        self._writer = threading.Thread(target=self._writer_loop, name="event-store", daemon=True)
        self._writer.start()

    # ------------------------------------------------------------------ intake
    def on_event(self, ev: dict, file: Path | None = None) -> None:
        """events.subscribe callback - runs on the agent's thread, must return at once.
        `file` names the call file when importing offline (live: taken from events.call_file())."""
        if not ev.get("corr"):
            return
        item = {**_keys(ev), "corr": ev["corr"], "seq": ev.get("seq"), "ts": ev.get("ts"), "t": ev.get("t"),
                "hop": ev.get("hop"), "kind": ev.get("kind"), "ms": ev.get("ms"),
                "text": _searchable(ev), "item": ev}
        self._q.put(item)
        meta = self._track_meta(ev, file)
        if meta is not None:
            self._q.put(meta)

    def _track_meta(self, ev: dict, file: Path | None = None) -> dict | None:
        """Build the per-call META item from the events as they pass by."""
        corr, k, p = ev["corr"], ev["kind"], ev.get("payload") or {}
        m = self._meta.get(corr)
        if k == "call.start":
            m = self._meta[corr] = {"corr": corr, "started": ev.get("ts"), "caller": p.get("sip_from", ""),
                                    "language": p.get("language"), "turns": 0, "gate": "IDLE",
                                    "worm_records": [], "cards_blocked": [], "transcript": [],
                                    "events": 0, "outcome": "in progress", "file": None}
            f = file
            if f is None:
                from agent import events as _ev
                f = _ev.call_file()
            if f is not None:
                m["file"] = f.name
                m["audio"] = f.with_suffix(".wav").name
        if m is None:
            return None
        m["events"] += 1
        if k == "stt.done" and p.get("text"):
            m["transcript"].append({"who": "caller", "t": ev.get("t"), "text": p["text"]})
        elif k == "llm.sentence" and p.get("text"):
            m["transcript"].append({"who": "agent", "t": ev.get("t"), "text": p["text"]})
        elif k == "tts.greeting":
            m["transcript"].append({"who": "agent", "t": ev.get("t"), "text": p.get("text", "")})
        elif k == "vad.utterance":
            m["turns"] = max(m["turns"], int(p.get("turn") or 0))
        elif k == "gate.state":
            m["gate"] = p.get("to")
        elif k == "worm.ack":
            m["worm_records"].append(p.get("record_id"))
        elif k == "bank.card_blocked":
            m["cards_blocked"].append(p.get("card"))
        elif k == "speech.switch":
            m["language"] = p.get("to")
        elif k == "call.end":
            m["ended"] = ev.get("ts")
            m["duration_s"] = round((ev.get("ts") or 0) - (m.get("started") or 0), 1)
            m["reason"] = p.get("reason")
            m["outcome"] = ("card blocked" if m["cards_blocked"] else
                            "escalated" if m["gate"] == "FROZEN" else "no action")
            self._meta.pop(corr, None)
        ts = float(m.get("started") or ev.get("ts") or time.time())
        day = time.strftime("%Y%m%d", time.localtime(ts))
        return {"PK": f"CALL#{corr}", "SK": "META", "GSI1PK": "META", "GSI1SK": ts,
                "GSI2PK": f"DAY#{day}", "GSI2SK": ts, "corr": corr, "seq": None, "ts": m.get("started"),
                "t": None, "hop": None, "kind": "META", "ms": None,
                "text": " ".join(x["text"] for x in m["transcript"][-40:]) + " " + m["caller"],
                "item": {**m, "transcript": m["transcript"][-200:]}}

    def _writer_loop(self):
        while True:
            batch = [self._q.get()]
            deadline = time.time() + 0.2
            while time.time() < deadline:
                try:
                    batch.append(self._q.get(timeout=0.05))
                except queue.Empty:
                    break
            try:
                self._put_items(batch)
            except Exception as e:  # noqa: BLE001 - the store is best effort, never the agent's problem
                print(f"[event-store] write failed: {e}")

    def flush(self, timeout: float = 3.0) -> None:
        t0 = time.time()
        while not self._q.empty() and time.time() - t0 < timeout:
            time.sleep(0.05)
        time.sleep(0.3)

    # ------------------------------------------------------------------ to implement
    def _put_items(self, items: list[dict]) -> None:
        raise NotImplementedError

    def list_calls(self, limit: int = 50, day: str | None = None) -> list[dict]:
        raise NotImplementedError

    def get_call(self, corr: str) -> dict | None:
        raise NotImplementedError

    def query_events(self, corr: str, kind: str | None = None, hop: str | None = None,
                     limit: int = 5000) -> list[dict]:
        raise NotImplementedError

    def search(self, text: str = "", kind: str | None = None, hop: str | None = None,
               corr: str | None = None, limit: int = 100) -> dict:
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError


# ====================================================================== SQLite
class SqliteStore(EventStore):
    name = "sqlite"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS items (
                pk TEXT NOT NULL, sk TEXT NOT NULL,
                gsi1pk TEXT, gsi1sk REAL, gsi2pk TEXT, gsi2sk REAL,
                corr TEXT, seq INTEGER, ts REAL, t REAL, hop TEXT, kind TEXT, ms REAL,
                text TEXT, item TEXT NOT NULL,
                PRIMARY KEY (pk, sk)
            );
            CREATE INDEX IF NOT EXISTS gsi1 ON items (gsi1pk, gsi1sk);
            CREATE INDEX IF NOT EXISTS gsi2 ON items (gsi2pk, gsi2sk);
            CREATE INDEX IF NOT EXISTS by_kind_corr ON items (corr, kind);
            CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(text, pk UNINDEXED, sk UNINDEXED,
                tokenize = 'unicode61 remove_diacritics 2');
        """)
        super().__init__()

    def _put_items(self, items: list[dict]) -> None:
        with self._lock:
            for it in items:
                self._db.execute(
                    "INSERT OR REPLACE INTO items (pk,sk,gsi1pk,gsi1sk,gsi2pk,gsi2sk,corr,seq,ts,t,hop,kind,ms,text,item) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (it["PK"], it["SK"], it["GSI1PK"], it["GSI1SK"], it["GSI2PK"], it["GSI2SK"], it["corr"],
                     it["seq"], it["ts"], it["t"], it["hop"], it["kind"], it["ms"], it["text"],
                     json.dumps(it["item"], ensure_ascii=False)))
                self._db.execute("DELETE FROM items_fts WHERE pk=? AND sk=?", (it["PK"], it["SK"]))
                self._db.execute("INSERT INTO items_fts (text, pk, sk) VALUES (?,?,?)", (it["text"], it["PK"], it["SK"]))
            self._db.commit()

    @staticmethod
    def _row(r) -> dict:
        return json.loads(r[0])

    def list_calls(self, limit: int = 50, day: str | None = None) -> list[dict]:
        with self._lock:
            if day:
                rows = self._db.execute("SELECT item FROM items WHERE sk='META' AND gsi2pk=? ORDER BY gsi2sk DESC LIMIT ?",
                                        (f"DAY#{day}", limit)).fetchall()
            else:
                rows = self._db.execute("SELECT item FROM items WHERE sk='META' ORDER BY gsi1sk DESC LIMIT ?",
                                        (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def get_call(self, corr: str) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT item FROM items WHERE pk=? AND sk='META'", (f"CALL#{corr}",)).fetchone()
        return self._row(r) if r else None

    def query_events(self, corr: str, kind: str | None = None, hop: str | None = None, limit: int = 5000) -> list[dict]:
        sql, args = "SELECT item FROM items WHERE pk=? AND sk LIKE 'EV#%'", [f"CALL#{corr}"]
        if kind:
            sql += " AND kind=?"; args.append(kind)
        if hop:
            sql += " AND hop=?"; args.append(hop)
        sql += " ORDER BY sk LIMIT ?"; args.append(limit)
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [self._row(r) for r in rows]

    def search(self, text: str = "", kind: str | None = None, hop: str | None = None,
               corr: str | None = None, limit: int = 100) -> dict:
        args: list = []
        if text.strip():
            sql = ("SELECT i.item, bm25(items_fts) AS rank FROM items_fts JOIN items i ON i.pk=items_fts.pk AND i.sk=items_fts.sk "
                   "WHERE items_fts MATCH ?")
            q = " ".join(f'"{w}"*' for w in text.split())    # phrase-safe prefix terms ("worm-0000"* finds WORM-000017)
            args.append(q)
        else:
            sql = "SELECT i.item, 0 AS rank FROM items i WHERE 1=1"
        if kind:
            sql += " AND i.kind=?"; args.append(kind)
        else:
            sql += " AND i.sk != 'META'"
        if hop:
            sql += " AND i.hop=?"; args.append(hop)
        if corr:
            sql += " AND i.corr=?"; args.append(corr)
        sql += " ORDER BY rank, i.ts DESC LIMIT ?"; args.append(limit)
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return {"backend": self.name, "full_text": True, "count": len(rows), "results": [self._row(r) for r in rows]}

    def stats(self) -> dict:
        with self._lock:
            n = self._db.execute("SELECT COUNT(*) FROM items WHERE sk != 'META'").fetchone()[0]
            c = self._db.execute("SELECT COUNT(*) FROM items WHERE sk = 'META'").fetchone()[0]
            kinds = self._db.execute("SELECT kind, COUNT(*) FROM items WHERE sk != 'META' GROUP BY kind ORDER BY 2 DESC LIMIT 40").fetchall()
        return {"backend": self.name, "path": str(self.path), "events": n, "calls": c,
                "size_mb": round(self.path.stat().st_size / 1e6, 2) if self.path.exists() else 0,
                "kinds": [{"kind": k, "n": v} for k, v in kinds]}


# ====================================================================== DynamoDB (AWS or DynamoDB Local)
class DynamoStore(EventStore):
    name = "dynamodb"

    def __init__(self, table: str, region: str = "eu-central-1", endpoint_url: str | None = None, create: bool = True):
        import boto3
        from boto3.dynamodb.conditions import Key, Attr
        self._Key, self._Attr = Key, Attr
        self._res = boto3.resource("dynamodb", region_name=region, endpoint_url=endpoint_url or None)
        self._table_name = table
        if create:
            self._ensure_table()
        self.table = self._res.Table(table)
        super().__init__()

    def _ensure_table(self):
        names = [t.name for t in self._res.tables.all()]
        if self._table_name in names:
            return
        self._res.create_table(
            TableName=self._table_name,
            KeySchema=[{"AttributeName": "PK", "KeyType": "HASH"}, {"AttributeName": "SK", "KeyType": "RANGE"}],
            AttributeDefinitions=[{"AttributeName": "PK", "AttributeType": "S"}, {"AttributeName": "SK", "AttributeType": "S"},
                                  {"AttributeName": "GSI1PK", "AttributeType": "S"}, {"AttributeName": "GSI1SK", "AttributeType": "N"},
                                  {"AttributeName": "GSI2PK", "AttributeType": "S"}, {"AttributeName": "GSI2SK", "AttributeType": "N"}],
            GlobalSecondaryIndexes=[
                {"IndexName": "GSI1", "KeySchema": [{"AttributeName": "GSI1PK", "KeyType": "HASH"}, {"AttributeName": "GSI1SK", "KeyType": "RANGE"}],
                 "Projection": {"ProjectionType": "ALL"}},
                {"IndexName": "GSI2", "KeySchema": [{"AttributeName": "GSI2PK", "KeyType": "HASH"}, {"AttributeName": "GSI2SK", "KeyType": "RANGE"}],
                 "Projection": {"ProjectionType": "ALL"}}],
            BillingMode="PAY_PER_REQUEST").wait_until_exists()

    @staticmethod
    def _dyn(v):
        """DynamoDB wants Decimal for numbers and no empty strings inside maps: store the event as JSON text."""
        from decimal import Decimal
        return Decimal(str(v)) if isinstance(v, float) else v

    def _put_items(self, items: list[dict]) -> None:
        with self.table.batch_writer() as bw:
            for it in items:
                bw.put_item(Item={k: self._dyn(v) for k, v in it.items() if k != "item" and v is not None}
                            | {"item": json.dumps(it["item"], ensure_ascii=False)})

    @staticmethod
    def _row(it) -> dict:
        return json.loads(it["item"])

    def list_calls(self, limit: int = 50, day: str | None = None) -> list[dict]:
        if day:
            r = self.table.query(IndexName="GSI2", KeyConditionExpression=self._Key("GSI2PK").eq(f"DAY#{day}"),
                                 FilterExpression=self._Attr("SK").eq("META"), ScanIndexForward=False, Limit=limit)
        else:
            r = self.table.query(IndexName="GSI1", KeyConditionExpression=self._Key("GSI1PK").eq("META"),
                                 ScanIndexForward=False, Limit=limit)
        return [self._row(i) for i in r.get("Items", [])]

    def get_call(self, corr: str) -> dict | None:
        r = self.table.get_item(Key={"PK": f"CALL#{corr}", "SK": "META"})
        return self._row(r["Item"]) if "Item" in r else None

    def query_events(self, corr: str, kind: str | None = None, hop: str | None = None, limit: int = 5000) -> list[dict]:
        kw = {"KeyConditionExpression": self._Key("PK").eq(f"CALL#{corr}") & self._Key("SK").begins_with("EV#"), "Limit": limit}
        f = None
        if kind:
            f = self._Attr("kind").eq(kind)
        if hop:
            f = (f & self._Attr("hop").eq(hop)) if f is not None else self._Attr("hop").eq(hop)
        if f is not None:
            kw["FilterExpression"] = f
        return [self._row(i) for i in self.table.query(**kw).get("Items", [])]

    def search(self, text: str = "", kind: str | None = None, hop: str | None = None,
               corr: str | None = None, limit: int = 100) -> dict:
        f = None
        for word in text.split():
            c = self._Attr("text").contains(word)
            f = (f & c) if f is not None else c
        if hop:
            c = self._Attr("hop").eq(hop); f = (f & c) if f is not None else c
        if kind:
            r = self.table.query(IndexName="GSI1", KeyConditionExpression=self._Key("GSI1PK").eq(f"KIND#{kind}"),
                                 ScanIndexForward=False, Limit=limit, **({"FilterExpression": f} if f is not None else {}))
        elif corr:
            r = self.table.query(KeyConditionExpression=self._Key("PK").eq(f"CALL#{corr}"), Limit=limit,
                                 **({"FilterExpression": f} if f is not None else {}))
        else:
            r = self.table.scan(Limit=max(limit, 1000), **({"FilterExpression": f} if f is not None else {}))
        items = [self._row(i) for i in r.get("Items", []) if i.get("SK") != "META"][:limit]
        return {"backend": self.name, "full_text": False, "count": len(items), "results": items,
                "note": "DynamoDB: substring filter on the text attribute, no ranking"}

    def stats(self) -> dict:
        d = self.table.meta.client.describe_table(TableName=self._table_name)["Table"]
        return {"backend": self.name, "table": self._table_name, "items_estimate": d.get("ItemCount"),
                "size_mb": round((d.get("TableSizeBytes") or 0) / 1e6, 2)}


# ====================================================================== factory + import
def make_store(settings) -> EventStore:
    if settings.STORE_BACKEND == "dynamodb":
        return DynamoStore(settings.DYNAMODB_TABLE, settings.DYNAMODB_REGION, settings.DYNAMODB_ENDPOINT or None)
    return SqliteStore(PROJECT / settings.SQLITE_PATH)


def import_files(store: EventStore, files=None) -> int:
    """Load existing calls\\*.jsonl into the store (idempotent - same keys overwrite)."""
    import re
    files = files or sorted(p for p in CALLS.glob("*.jsonl") if re.match(r"\d{8}-\d{6}_", p.name))
    n = 0
    for p in files:
        evs = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ev in evs:
            store.on_event(ev, file=p)
            n += 1
    store.flush(timeout=30)
    return n


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT))
    from agent import settings
    st = make_store(settings)
    if "--import" in sys.argv:
        n = import_files(st)
        print(f"imported {n} events into {st.name}")
    print(json.dumps(st.stats(), indent=2))
    if "--search" in sys.argv:
        q = sys.argv[sys.argv.index("--search") + 1]
        for r in st.search(q, limit=20)["results"]:
            print(f"{r.get('corr','')[:8]} {r.get('t')} {r.get('hop')} {r.get('kind')} {json.dumps(r.get('payload'), ensure_ascii=False)[:100]}")
