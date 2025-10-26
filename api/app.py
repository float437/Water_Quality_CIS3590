from __future__ import annotations
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ServerSelectionTimeoutError, PyMongoError
from dotenv import load_dotenv


# ------------------------------------------------------------
# Load .env from project root
#   - Expects: MONGO_USER, MONGO_PASS, MONGO_CLUSTER_URL
#   - Optional: DB_NAME, COLL_NAME
# ------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")
MONGO_CLUSTER_URL = os.getenv("MONGO_CLUSTER_URL")
DB_NAME = os.getenv("DB_NAME", "water_quality_data")
COLL_NAME = os.getenv("COLL_NAME", "asv_1")


# ------------------------------------------------------------
# Build connection URI (Atlas or Local)
#   - Uses SRV + TLS for Atlas when creds are present
#   - Falls back to local mongodb://localhost:27017 otherwise
# ------------------------------------------------------------
if MONGO_USER and MONGO_PASS and MONGO_CLUSTER_URL:
    user = quote_plus(MONGO_USER)
    pwd = quote_plus(MONGO_PASS)
    MONGODB_URI = f"mongodb+srv://{user}:{pwd}@{MONGO_CLUSTER_URL}/?retryWrites=true&w=majority&tls=true"
else:
    MONGODB_URI = "mongodb://localhost:27017"

# ------------------------------------------------------------
# DB client w/ fallback to mongomock + auto-seed from cleaned CSVs
# ------------------------------------------------------------
DB_NAME = os.getenv("DB_NAME", "water_quality_data")
COLL_NAME = os.getenv("COLL_NAME", "asv_1")

def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None

def _normalize_row(r: dict) -> dict:
    out = {}
    # timestamp
    for k in ["timestamp", "Timestamp", "DateTime", "Date", "Time", "Date m/d/y   "]:
        if k in r and pd.notna(r[k]):
            try:
                out["timestamp"] = pd.to_datetime(r[k]).isoformat()
                break
            except Exception:
                pass
    # lat/lon
    if "Latitude" in r:  out["latitude"]  = _to_float(r["Latitude"])
    if "Longitude" in r: out["longitude"] = _to_float(r["Longitude"])
    # numeric sensors
    for src, dst in [
        ("temperature","temperature"), ("Temperature (c)","temperature"), ("Temp C","temperature"),
        ("salinity","salinity"), ("Salinity (ppt)","salinity"), ("Sal ppt","salinity"),
        ("odo","odo"), ("ODO mg/L","odo"), ("ODOsat %","odo"),
    ]:
        if src in r and pd.notna(r[src]):
            val = _to_float(r[src])
            if val is not None:
                out[dst] = val
    return out

def _seed_if_empty(col):
    if col.estimated_document_count() > 0:
        return
    candidates = [
        "data/Clean_CSV/cleanedOctober7.csv",
        "data/Clean_CSV/cleanedOctober21.csv",
        "data/Clean_CSV/cleanedNovember16.csv",
        "data/Clean_CSV/cleanedDecember16.csv",
        "Clean_CSV/cleanedOctober7.csv",
        "Clean_CSV/cleanedOctober21.csv",
        "Clean_CSV/cleanedNovember16.csv",
        "Clean_CSV/cleanedDecember16.csv",
    ]
    total = 0
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                docs = [_normalize_row(rec) for rec in df.to_dict(orient="records")]
                docs = [d for d in docs if d.get("timestamp")]
                if docs:
                    col.insert_many(docs, ordered=False)
                    total += len(docs)
                    print(f"[seed] Loaded {len(docs)} rows from {path}")
            except Exception as e:
                print(f"[seed] Failed {path}: {e}")
    print(f"[seed] Total loaded: {total}")

def _get_db_and_col():
    from pymongo import MongoClient
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=4000)
        client.admin.command("ping")
        print("[✅ MongoDB] Connected:", MONGODB_URI)
        return client[DB_NAME], COLL_NAME, False  # not mock
    except Exception as e:
        print(f"[ℹ️ MongoDB] Not available, falling back to mongomock: {e}")
        import mongomock
        client = mongomock.MongoClient()
        print("[➡️  Using mongomock (in-memory)]")
        return client[DB_NAME], COLL_NAME, True
    
app = Flask(__name__)

db, _COLL, _IS_MOCK = _get_db_and_col()
col = db[_COLL]
if _IS_MOCK:
    _seed_if_empty(col)

# =======================
# Helpers for query build
# =======================
def _query_from_args(args):
    q = {}
    # time range (note: timestamps stored as ISO strings; we filter with ISO strings)
    start = args.get("start")
    end   = args.get("end")
    if start or end:
        t = {}
        if start:
            t["$gte"] = pd.to_datetime(start).isoformat()
        if end:
            t["$lte"] = pd.to_datetime(end).isoformat()
        q["timestamp"] = t

    def add_range(field, kmin, kmax):
        lo = args.get(kmin, None)
        hi = args.get(kmax, None)
        r = {}
        if lo is not None and lo != "":
            r["$gte"] = float(lo)
        if hi is not None and hi != "":
            r["$lte"] = float(hi)
        if r:
            q[field] = r

    add_range("temperature", "min_temp", "max_temp")
    add_range("salinity",    "min_sal",  "max_sal")
    add_range("odo",         "min_odo",  "max_odo")
    return q

def _to_df(docs):
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    # coerce timestamp + numerics
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["temperature", "salinity", "odo", "latitude", "longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _df_to_items(df):
    if df.empty:
        return []
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = out["timestamp"].dt.tz_localize(None).dt.isoformat()
    # keep only known fields if you prefer:
    # fields = ["timestamp","temperature","salinity","odo","latitude","longitude"]
    # out = out[ [c for c in fields if c in out.columns] ]
    return out.where(pd.notnull(out), None).to_dict(orient="records")


# ===========
# Endpoints
# ===========

@app.get("/api/observations")
def observations():
    try:
        limit = int(request.args.get("limit", 100))
        limit = max(1, min(limit, 1000))
        skip  = int(request.args.get("skip", 0))

        query = _query_from_args(request.args)
        total = col.count_documents(query)

        cursor = (
            col.find(query, {"_id": 0})
               .sort("timestamp", ASCENDING)
               .skip(skip)
               .limit(limit)
        )
        items = list(cursor)
        return jsonify({"count": total, "items": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/stats")
def stats():
    try:
        query = _query_from_args(request.args)
        docs = list(col.find(query, {"_id": 0}))
        df = _to_df(docs)

        result = {}
        for field in ["temperature", "salinity", "odo"]:
            if field in df.columns:
                s = pd.to_numeric(df[field], errors="coerce").dropna()
                if s.empty:
                    result[field] = {"count": 0}
                else:
                    p25, p50, p75 = np.percentile(s, [25, 50, 75])
                    result[field] = {
                        "count": int(s.count()),
                        "mean":  float(s.mean()),
                        "min":   float(s.min()),
                        "max":   float(s.max()),
                        "p25":   float(p25),
                        "p50":   float(p50),
                        "p75":   float(p75),
                    }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/outliers")
def outliers():
    try:
        field  = request.args.get("field", "temperature")
        method = request.args.get("method", "iqr").lower()
        k      = float(request.args.get("k", 1.5))

        if field not in ["temperature", "salinity", "odo"]:
            return jsonify({"error": "field must be one of temperature, salinity, odo"}), 400

        query = _query_from_args(request.args)
        docs = list(col.find(query, {"_id": 0}))
        df = _to_df(docs)

        if df.empty or field not in df.columns:
            return jsonify([])

        s = pd.to_numeric(df[field], errors="coerce")
        dff = df.copy()
        dff[field] = s
        dff = dff.dropna(subset=[field])

        if dff.empty:
            return jsonify([])

        if method == "zscore":
            mu = dff[field].mean()
            sd = dff[field].std(ddof=0) or 1.0
            flagged = dff[(dff[field] - mu).abs() / sd > k]
        else:
            q1, q3 = dff[field].quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - k * iqr, q3 + k * iqr
            flagged = dff[(dff[field] < lo) | (dff[field] > hi)]

        return jsonify(_df_to_items(flagged))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Initialize Flask + MongoDB Client with Error Handling
#   - Performs a ping on startup to validate connectivity
#   - If unreachable, keeps coll=None so routes can return 503 gracefully
# ------------------------------------------------------------
app = Flask(__name__)

try:
    client = MongoClient(MONGODB_URI, tz_aware=True, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    print("[✅ MongoDB] Connection successful.")
    db = client[DB_NAME]
    coll = db[COLL_NAME]

    # Optional index for time-range queries; safe if already exists
    try:
        coll.create_index([("timestamp", ASCENDING)])
    except Exception:
        pass
except ServerSelectionTimeoutError as e:
    print("[❌ MongoDB] Cannot connect to cluster:", e)
    coll = None
except PyMongoError as e:
    print("[❌ MongoDB] General connection error:", e)
    coll = None


# ------------------------------------------------------------
# Utility Functions
#   - Helpers for parsing times, pagination clamping, query building,
#     document shaping, and stats summarization.
# ------------------------------------------------------------
def parse_iso(dt_str: str) -> datetime:
    """
    Parse an ISO8601 timestamp string (accepts trailing 'Z') into an
    aware UTC datetime instance.
    Raises ValueError if the input cannot be parsed.
    """
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_iso_z(dt: datetime) -> str:
    """
    Convert a datetime (naive or aware) to ISO8601 string in UTC with 'Z'.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def clamp_limit(s: str | None, default=100, max_allowed=1000) -> int:
    """
    Clamp the 'limit' query parameter to a sane range:
      - default if missing or invalid
      - minimum 1
      - maximum max_allowed (1000)
    """
    if s is None:
        return default
    try:
        n = int(s)
        if n < 1:
            return default
        return min(n, max_allowed)
    except Exception:
        return default


def build_observation_query():
    """
    Build the MongoDB filter from request query params.
    Supports:
      - start / end (ISO timestamps)
      - min_temp / max_temp
      - min_sal  / max_sal
      - min_odo  / max_odo
      - limit (default 100, max 1000)
      - skip  (>= 0)
    Returns: (filter_dict, limit:int, skip:int)
    """
    q = {}
    # Time filters
    start = request.args.get("start")
    end = request.args.get("end")
    if start or end:
        q["timestamp"] = {}
        try:
            if start:
                q["timestamp"]["$gte"] = parse_iso(start)
            if end:
                q["timestamp"]["$lte"] = parse_iso(end)
        except Exception:
            raise ValueError("Invalid start/end timestamp. Use ISO8601 format (e.g., 2025-10-26T12:00:00Z).")

    # Numeric ranges
    def add_range(param_base, field):
        min_s = request.args.get(f"min_{param_base}")
        max_s = request.args.get(f"max_{param_base}")
        if min_s or max_s:
            q[field] = {}
        if min_s:
            q[field]["$gte"] = float(min_s)
        if max_s:
            q[field]["$lte"] = float(max_s)

    add_range("temp", "temperature")
    add_range("sal", "salinity")
    add_range("odo", "odo")

    # Pagination
    limit = clamp_limit(request.args.get("limit"))
    try:
        skip = max(0, int(request.args.get("skip", "0")))
    except Exception:
        skip = 0

    return q, limit, skip


def doc_to_public(d):
    """
    Map a MongoDB document to the public API shape.
    Includes geo fields if present (latitude/longitude).
    """
    ts = d.get("timestamp")
    return {
        "timestamp": to_iso_z(ts) if isinstance(ts, datetime) else ts,
        "temperature": d.get("temperature"),
        "salinity": d.get("salinity"),
        "odo": d.get("odo"),
        "latitude": d.get("latitude"),
        "longitude": d.get("longitude"),
    }


def summarize(values):
    """
    Compute summary statistics for a numeric list (may contain None):
      - count, mean, min, max, p25, p50, p75
    Uses NumPy percentiles with method='linear' (NumPy 2.x).
    """
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"count": 0, "mean": None, "min": None, "max": None, "p25": None, "p50": None, "p75": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25, method="linear")),
        "p50": float(np.percentile(arr, 50, method="linear")),
        "p75": float(np.percentile(arr, 75, method="linear")),
    }


# ------------------------------------------------------------
# Routes (Endpoints)
# ------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """
    Health Check Endpoint
    GET /api/health
    Returns a simple JSON payload confirming the API is up.
    Response: { "status": "ok" }
    """
    return jsonify({"status": "ok"}), 200


@app.get("/api/observations")
def get_observations():
    """
    Observations Endpoint
    GET /api/observations

    Returns documents with optional filters and pagination:
      - start / end (ISO timestamps)
      - min_temp / max_temp
      - min_sal  / max_sal
      - min_odo  / max_odo
      - limit (default 100, max 1000)
      - skip  (for pagination)

    Response shape:
    {
      "count": <total after filters>,
      "limit": <limit>,
      "skip": <skip>,
      "items": [
        {"timestamp": "...", "temperature": 27.2, "salinity": 35.1, "odo": 6.7, ...},
        ...
      ]
    }
    """
    # DB availability guard (returns 503 instead of crashing)
    if coll is None:
        return jsonify({"error": "MongoDB connection not available. Check credentials or network."}), 503

    try:
        q, limit, skip = build_observation_query()
        total = coll.count_documents(q)

        projection = {
            "_id": False,
            "timestamp": True, "temperature": True, "salinity": True, "odo": True,
            "latitude": True, "longitude": True,
        }

        cursor = (
            coll.find(q, projection=projection)
                .sort("timestamp", ASCENDING)
                .skip(skip)
                .limit(limit)
        )
        items = [doc_to_public(doc) for doc in cursor]
        return jsonify({"count": total, "limit": limit, "skip": skip, "items": items}), 200

    except ServerSelectionTimeoutError:
        return jsonify({"error": "MongoDB server not reachable (timeout)."}), 503
    except PyMongoError as e:
        return jsonify({"error": f"Database error: {type(e).__name__}"}), 500
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400


@app.get("/api/stats")
def get_stats():
    """
    Statistics Endpoint
    GET /api/stats

    Returns summary statistics for numeric fields (temperature, salinity, odo):
      - count, mean, min, max, p25, p50, p75

    Supports the SAME optional filters as /api/observations (except limit/skip).
    """
    # DB availability guard (returns 503 instead of crashing)
    if coll is None:
        return jsonify({"error": "MongoDB connection not available. Check credentials or network."}), 503

    try:
        q, _, _ = build_observation_query()
        proj = {"_id": False, "temperature": True, "salinity": True, "odo": True}
        docs = list(coll.find(q, projection=proj))

        temps = [d.get("temperature") for d in docs]
        sals  = [d.get("salinity") for d in docs]
        odos  = [d.get("odo") for d in docs]

        return jsonify({
            "temperature": summarize(temps),
            "salinity": summarize(sals),
            "odo": summarize(odos),
        }), 200

    except ServerSelectionTimeoutError:
        return jsonify({"error": "MongoDB server not reachable (timeout)."}), 503
    except PyMongoError as e:
        return jsonify({"error": f"Database error: {type(e).__name__}"}), 500
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400


@app.get("/api/outliers")
def get_outliers():
    """
    Outliers Endpoint
    GET /api/outliers?field=temperature&method=iqr&k=1.5

    Detect outliers on demand using IQR or z-score for the selected field.
    Query params:
      - field  = temperature | salinity | odo   (required)
      - method = iqr | z-score | z | zscore     (default: iqr)
      - k      = float (IQR multiplier or z threshold; default 1.5 for iqr, 3.0 for z)

    Supports the SAME filters as /api/observations to scope the population:
      - start, end, min_temp/max_temp, min_sal/max_sal, min_odo/max_odo

    Returns:
    {
      "field": "<field>",
      "method": "iqr" | "z-score",
      "k": <float>,
      "thresholds": { ... },   # e.g., {"lower": L, "upper": U} for IQR or {"mean": M, "std": S, "k": K} for z
      "items": [ <flagged documents in public shape> ]
    }
    """
    # DB availability guard (returns 503 instead of crashing)
    if coll is None:
        return jsonify({"error": "MongoDB connection not available. Check credentials or network."}), 503

    # ---- validate inputs
    field = (request.args.get("field") or "").strip()
    if field not in {"temperature", "salinity", "odo"}:
        return jsonify({"error": "Missing or invalid 'field'. Must be one of: temperature, salinity, odo."}), 400

    method_raw = (request.args.get("method") or "iqr").strip().lower()
    method = "iqr" if method_raw == "iqr" else ("z-score" if method_raw in {"z", "zscore", "z-score"} else None)
    if method is None:
        return jsonify({"error": "Invalid 'method'. Use 'iqr' or 'z-score'."}), 400

    k_param = request.args.get("k")
    try:
        k = float(k_param) if k_param is not None else (1.5 if method == "iqr" else 3.0)
    except Exception:
        return jsonify({"error": "Parameter 'k' must be numeric."}), 400

    try:
        # Reuse filters; ignore limit/skip for population-wide calc
        q, _, _ = build_observation_query()

        # Pull only what's needed; keep stable order by timestamp
        projection = {
            "_id": False, "timestamp": True,
            "temperature": True, "salinity": True, "odo": True,
            "latitude": True, "longitude": True
        }
        rows = list(coll.find(q, projection=projection).sort("timestamp", ASCENDING))

        # Collect (doc, value) pairs for the selected field
        pairs = [(doc, doc.get(field)) for doc in rows if isinstance(doc.get(field), (int, float, float))]
        if not pairs:
            return jsonify({"field": field, "method": method, "k": k, "thresholds": None, "items": []}), 200

        # Vector of values
        import numpy as np  # (kept here as in the provided snippet)
        vals = np.array([v for _, v in pairs], dtype=float)

        flagged = np.zeros(vals.size, dtype=bool)
        thresholds = {}

        if method == "iqr":
            # NumPy 2.x uses method="linear" for classic linear interpolation
            q1 = float(np.percentile(vals, 25, method="linear"))
            q3 = float(np.percentile(vals, 75, method="linear"))
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            thresholds = {"lower": lower, "upper": upper}
            flagged = (vals < lower) | (vals > upper)

        else:  # z-score
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))  # population std
            thresholds = {"mean": mean, "std": std, "k": k}
            if std > 0.0:
                z = np.abs((vals - mean) / std)
                flagged = z > k
            else:
                flagged = np.zeros_like(vals, dtype=bool)

        # Build flagged items in the same time-sorted order
        flagged_items = [doc_to_public(pairs[i][0]) for i in range(len(pairs)) if flagged[i]]

        return jsonify({
            "field": field,
            "method": method,
            "k": k,
            "thresholds": thresholds,
            "items": flagged_items
        }), 200

    except ServerSelectionTimeoutError:
        return jsonify({"error": "MongoDB server not reachable (timeout)."}), 503
    except PyMongoError as e:
        return jsonify({"error": f"Database error: {type(e).__name__}"}), 500
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400


# ------------------------------------------------------------
# Run App
# ------------------------------------------------------------
if __name__ == "__main__":
    # Development server (override host/port via env if needed)
    app.run(host="0.0.0.0", port=5000, debug=True)
