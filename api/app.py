from __future__ import annotations
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient, ASCENDING

# ------------------ Flask App ------------------
app = Flask(__name__)
CORS(app)

# ------------------ Mongo + fallback ------------------
DB_NAME = "water_quality_data"
COLL_NAME = "asv_1"

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

    # numeric mapping
    if "Latitude" in r:
        out["latitude"] = _to_float(r["Latitude"])
    if "Longitude" in r:
        out["longitude"] = _to_float(r["Longitude"])

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
                docs = [_normalize_row(x) for x in df.to_dict(orient="records")]
                docs = [d for d in docs if d.get("timestamp")]
                if docs:
                    col.insert_many(docs, ordered=False)
                    total += len(docs)
            except Exception:
                pass
    print(f"[seed] Total loaded: {total}")

def _get_db_and_col():
    try:
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=4000)
        client.admin.command("ping")
        print("[✅ MongoDB] Connected local")
        return client[DB_NAME], COLL_NAME, False
    except Exception as e:
        print("[ℹ️ MongoDB] Not available — using mongomock:", e)
        import mongomock
        client = mongomock.MongoClient()
        return client[DB_NAME], COLL_NAME, True

db, _COLL, IS_MOCK = _get_db_and_col()
col = db[_COLL]
if IS_MOCK:
    _seed_if_empty(col)

# ------------------ INPUT VALIDATION ------------------
def _safe_float(v, name):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        raise ValueError(f"Invalid float for {name}")

def _safe_limit(v):
    try:
        n = int(v)
    except Exception:
        raise ValueError("limit must be integer")
    return max(1, min(n, 1000))

def _safe_skip(v):
    try:
        n = int(v)
    except Exception:
        raise ValueError("skip must be integer")
    return max(0, n)

# ------------------ Query helper ------------------
def _query_from_args(args):
    q = {}

    start = args.get("start")
    end = args.get("end")
    if start or end:
        t = {}
        if start:
            t["$gte"] = pd.to_datetime(start).isoformat()
        if end:
            t["$lte"] = pd.to_datetime(end).isoformat()
        q["timestamp"] = t

    def add_range(field, mn, mx):
        lo = args.get(mn)
        hi = args.get(mx)
        r = {}
        if lo not in (None, ""):
            r["$gte"] = float(lo)
        if hi not in (None, ""):
            r["$lte"] = float(hi)
        if r:
            q[field] = r

    add_range("temperature","min_temp","max_temp")
    add_range("salinity","min_sal","max_sal")
    add_range("odo","min_odo","max_odo")
    return q

def _to_df(docs):
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["temperature","salinity","odo","latitude","longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _df_to_items(df):
    if df.empty:
        return []
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = out["timestamp"].dt.tz_localize(None).dt.isoformat()
    return out.where(pd.notnull(out), None).to_dict(orient="records")

# ------------------ Routes ------------------
@app.get("/api/health")
def health():
    return jsonify({"status":"ok"}), 200

@app.get("/api/observations")
def observations():
    try:
        limit = _safe_limit(request.args.get("limit",100))
        skip  = _safe_skip(request.args.get("skip",0))

        query = _query_from_args(request.args)
        total = col.count_documents(query)
        items = list(
            col.find(query, {"_id":0}).sort("timestamp",ASCENDING).skip(skip).limit(limit)
        )
        return jsonify({"count":total, "items":items}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/stats")
def stats():
    try:
        query = _query_from_args(request.args)
        df = _to_df(list(col.find(query, {"_id":0})))
        result = {}
        for field in ["temperature","salinity","odo"]:
            if field in df.columns:
                s = df[field].dropna()
                if len(s)==0:
                    result[field] = {"count":0}
                else:
                    p25,p50,p75 = np.percentile(s,[25,50,75])
                    result[field] = {
                        "count": int(s.count()),
                        "mean": float(s.mean()),
                        "min": float(s.min()),
                        "max": float(s.max()),
                        "p25": float(p25),
                        "p50": float(p50),
                        "p75": float(p75)
                    }
        return jsonify(result),200
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.get("/api/outliers")
def outliers():
    try:
        field = request.args.get("field","temperature")
        method = request.args.get("method","iqr").lower()
        k = float(request.args.get("k",1.5))

        if field not in ["temperature","salinity","odo"]:
            return jsonify({"error":"field must be: temperature|salinity|odo"}),400

        query = _query_from_args(request.args)
        df = _to_df(list(col.find(query,{"_id":0})))
        if df.empty or field not in df.columns:
            return jsonify([])

        s = df[field].dropna()
        if method=="zscore":
            mu = s.mean()
            sd = s.std(ddof=0) or 1
            flagged = df[(s-mu).abs()/sd > k]
        else:
            q1,q3 = s.quantile([0.25,0.75])
            iqr = q3-q1
            lo,hi = q1-k*iqr, q3+k*iqr
            flagged = df[(s<lo)|(s>hi)]

        return jsonify(_df_to_items(flagged)),200
    except Exception as e:
        return jsonify({"error":str(e)}),500


# -------------- Extra Credit --------------
@app.get("/api/geo/box")
def geo_box():
    try:
        min_lat = _safe_float(request.args.get("min_lat"),"min_lat")
        max_lat = _safe_float(request.args.get("max_lat"),"max_lat")
        min_lon = _safe_float(request.args.get("min_lon"),"min_lon")
        max_lon = _safe_float(request.args.get("max_lon"),"max_lon")

        if None in (min_lat,max_lat,min_lon,max_lon):
            return jsonify({"error":"min_lat,max_lat,min_lon,max_lon required"}),400

        base = _query_from_args(request.args)
        base["latitude"] = {"$gte":min_lat,"$lte":max_lat}
        base["longitude"]= {"$gte":min_lon,"$lte":max_lon}
        
        limit = _safe_limit(request.args.get("limit",200))
        skip  = _safe_skip(request.args.get("skip",0))

        total = col.count_documents(base)
        items = list(
            col.find(base,{"_id":0})
              .sort("timestamp",ASCENDING)
              .skip(skip).limit(limit)
        )
        return jsonify({"count":total,"items":items}),200
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.get("/api/timeseries")
def timeseries():
    try:
        field = request.args.get("field","temperature")
        freq  = request.args.get("freq","5min")
        agg   = request.args.get("agg","mean").lower()

        if field not in ["temperature","salinity","odo"]:
            return jsonify({"error":"field must be: temperature|salinity|odo"}),400
        if agg not in ["mean","median","min","max"]:
            return jsonify({"error":"agg must be: mean|median|min|max"}),400

        df = _to_df(list(col.find(_query_from_args(request.args),{"_id":0})))
        if df.empty or "timestamp" not in df.columns:
            return jsonify([])

        df = df.dropna(subset=[field]).sort_values("timestamp")
        df = df.set_index("timestamp")

        if agg=="mean":
            s = df[field].resample(freq).mean()
        elif agg=="median":
            s = df[field].resample(freq).median()
        elif agg=="min":
            s = df[field].resample(freq).min()
        else:
            s = df[field].resample(freq).max()

        out = [{"timestamp":ts.isoformat(), field: (None if pd.isna(v) else float(v))}
               for ts,v in s.items()]
        return jsonify(out),200
    except Exception as e:
        return jsonify({"error":str(e)}),500
