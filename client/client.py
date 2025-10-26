import os
import math
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(page_title="Water Quality Client", layout="wide")
st.title("Water Quality — Streamlit Client")
st.caption("Streamlit client that consumes the Flask API")
DEFAULT_API = os.getenv("WQ_API_BASE", "http://127.0.0.1:5050").rstrip("/")

# ---------------------------------
# Sidebar: connection + filters
# ---------------------------------
with st.sidebar:
    st.header("Connection")
    api_base = st.text_input("API base URL", value=DEFAULT_API).rstrip("/")
    st.caption("Expected endpoints: /api/health, /api/observations, /api/stats, /api/outliers")

    st.divider()
    st.header("Filters")

    dr = st.date_input("Date range (optional)", value=[])
    start_iso = end_iso = ""
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        start_iso = datetime.combine(dr[0], datetime.min.time()).isoformat()
        end_iso   = datetime.combine(dr[1], datetime.max.time()).isoformat()

    col_min, col_max = st.columns(2)
    with col_min:
        min_temp = st.number_input("Min temperature", value=float("nan"))
        min_sal  = st.number_input("Min salinity",   value=float("nan"))
        min_odo  = st.number_input("Min ODO",        value=float("nan"))
    with col_max:
        max_temp = st.number_input("Max temperature", value=float("nan"))
        max_sal  = st.number_input("Max salinity",    value=float("nan"))
        max_odo  = st.number_input("Max ODO",         value=float("nan"))

    st.divider()
    st.header("Pagination")
    limit = st.number_input("Limit (max 1000)", min_value=1, max_value=1000, value=100, step=50)
    page  = st.number_input("Page", min_value=1, value=1, step=1)
    skip  = (page - 1) * limit

    if "_goto_page" in st.session_state:
        try:
            page = int(st.session_state["_goto_page"])
        except Exception:
            pass
        skip = (page - 1) * limit
        del st.session_state["_goto_page"]

    st.divider()
    st.header("Outliers")
    out_field  = st.selectbox("Field", ["temperature", "salinity", "odo"])
    out_method = st.selectbox("Method", ["iqr", "zscore"])
    out_k      = st.number_input("k (IQR multiplier or z threshold)", min_value=0.1, value=1.5, step=0.1)

# ---------------------------------
# Helpers
# ---------------------------------
def build_params():
    p = {"limit": limit, "skip": skip}
    if start_iso: p["start"] = start_iso
    if end_iso:   p["end"]   = end_iso
    if not math.isnan(min_temp): p["min_temp"] = min_temp
    if not math.isnan(max_temp): p["max_temp"] = max_temp
    if not math.isnan(min_sal):  p["min_sal"]  = min_sal
    if not math.isnan(max_sal):  p["max_sal"]  = max_sal
    if not math.isnan(min_odo):  p["min_odo"]  = min_odo
    if not math.isnan(max_odo):  p["max_odo"]  = max_odo
    return p

@st.cache_data(show_spinner=False)
def fetch_json(url: str, params: dict | None = None, timeout: int = 30):
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def df_from_items(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)

    for cand in ["timestamp", "datetime", "date_time", "Date", "Time"]:
        if cand in df.columns:
            df["timestamp"] = pd.to_datetime(df[cand], errors="coerce")
            break

    for c in ["temperature", "salinity", "odo", "latitude", "longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def api_ok() -> bool:
    try:
        h = fetch_json(f"{api_base}/api/health")
        return bool(h and h.get("status") == "ok")
    except Exception:
        return False

# ---------------------------------
# Health banner
# ---------------------------------
ok = api_ok()
st.info(f"API: `{api_base}` — Health: {'ok' if ok else 'unreachable'}")

# ---------------------------------
# Tabs per rubric
# ---------------------------------
tab_data, tab_charts, tab_stats, tab_outliers, tab_map = st.tabs(
    ["Data", "Charts", "Stats", "Outliers", "Map"]
)

# ---------------------------------
# Data tab
# ---------------------------------
with tab_data:
    st.subheader("Observations")

    if not ok:
        st.warning("API not reachable. Start your Flask API to populate this tab.")
    else:
        try:
            payload = fetch_json(f"{api_base}/api/observations", build_params())
            items = payload.get("items", payload if isinstance(payload, list) else [])
            count = payload.get("count", len(items))
            df = df_from_items(items)

            c1, c2, c3 = st.columns(3)
            c1.metric("Returned rows", len(df))
            c2.metric("Server-reported total", count)
            c3.metric("Page", page)

            st.dataframe(df, use_container_width=True, height=420)

            # Prev/Next
            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("⬅️ Prev") and page > 1:
                    st.session_state["_goto_page"] = page - 1
                    st.rerun()
            with col_next:
                if st.button("Next ➡️") and (skip + limit) < count:
                    st.session_state["_goto_page"] = page + 1
                    st.rerun()

            if not df.empty:
                st.download_button(
                    "Download CSV (this page)",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="observations_page.csv",
                    mime="text/csv",
                )
        except requests.HTTPError as e:
            st.error(f"HTTP error: {e.response.text}")
        except Exception as e:
            st.error(f"Error contacting /api/observations: {e}")

# ---------------------------------
# Charts tab (≥3 Plotly)
# ---------------------------------
with tab_charts:
    st.subheader("Visualizations")

    if not ok:
        st.warning("API not reachable. Start your Flask API to populate charts.")
    else:
        try:
            payload = fetch_json(f"{api_base}/api/observations", build_params())
            items = payload.get("items", payload if isinstance(payload, list) else [])
            df = df_from_items(items)

            if df.empty:
                st.warning("No data returned with current filters.")
            else:
                if "timestamp" in df.columns:
                    df = df.sort_values("timestamp")

                if {"timestamp", "temperature"}.issubset(df.columns):
                    st.markdown("**Temperature over time (line)**")
                    st.plotly_chart(px.line(df, x="timestamp", y="temperature", markers=True),
                                    use_container_width=True)

                if "salinity" in df.columns:
                    st.markdown("**Salinity distribution (histogram)**")
                    st.plotly_chart(px.histogram(df, x="salinity", nbins=30, marginal="box"),
                                    use_container_width=True)

                if {"temperature", "salinity"}.issubset(df.columns):
                    color_col = "odo" if "odo" in df.columns else None
                    st.markdown("**Temperature vs Salinity (scatter)**")
                    st.plotly_chart(px.scatter(df, x="temperature", y="salinity", color=color_col,
                                               hover_data=df.columns),
                                    use_container_width=True)
        except Exception as e:
            st.error(f"Error building charts: {e}")

# ---------------------------------
# Stats tab
# ---------------------------------
with tab_stats:
    st.subheader("Summary Statistics")

    if not ok:
        st.warning("API not reachable. Start your Flask API to populate stats.")
    else:
        try:
            stats = fetch_json(f"{api_base}/api/stats", build_params())
            if not stats:
                st.warning("No stats returned.")
            else:
                rows = []
                for field, d in stats.items():
                    row = {"field": field}
                    row.update(d)
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except requests.HTTPError as e:
            st.error(f"HTTP error: {e.response.text}")
        except Exception as e:
            st.error(f"Error contacting /api/stats: {e}")

# ---------------------------------
# Outliers tab
# ---------------------------------
with tab_outliers:
    st.subheader("Outliers")

    if not ok:
        st.warning("API not reachable. Start your Flask API to populate outliers.")
    else:
        try:
            params = build_params()
            params.update({"field": out_field, "method": out_method, "k": out_k})
            out = fetch_json(f"{api_base}/api/outliers", params)
            items = out if isinstance(out, list) else (out or {}).get("items", [])
            dfo = df_from_items(items)
            st.dataframe(dfo, use_container_width=True, height=420)
            st.info(f"Flagged records: {len(dfo)}")
            if not dfo.empty:
                st.download_button("Download Outliers CSV",
                                   dfo.to_csv(index=False).encode("utf-8"),
                                   "outliers.csv", "text/csv")
        except requests.HTTPError as e:
            st.error(f"HTTP error: {e.response.text}")
        except Exception as e:
            st.error(f"Error contacting /api/outliers: {e}")

# ---------------------------------
# Map tab
# ---------------------------------
with tab_map:
    st.subheader("Map (latitude / longitude)")

    if not ok:
        st.warning("API not reachable. Start your Flask API to populate the map.")
    else:
        try:
            payload = fetch_json(f"{api_base}/api/observations", build_params())
            items = payload.get("items", payload if isinstance(payload, list) else [])
            df = df_from_items(items)

            if df.empty or not {"latitude", "longitude"}.issubset(df.columns):
                st.warning("No latitude/longitude returned by the API to map.")
            else:
                mdf = df.dropna(subset=["latitude", "longitude"])
                if mdf.empty:
                    st.warning("No valid coordinates to plot.")
                else:
                    fig_map = px.scatter_mapbox(
                        mdf,
                        lat="latitude", lon="longitude",
                        hover_name="timestamp" if "timestamp" in mdf.columns else None,
                        hover_data=[c for c in ["temperature", "salinity", "odo"] if c in mdf.columns],
                        zoom=8, height=600,
                    )
                    fig_map.update_layout(mapbox_style="open-street-map",
                                          margin={"l": 0, "r": 0, "t": 0, "b": 0})
                    st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.error(f"Error building map: {e}")
