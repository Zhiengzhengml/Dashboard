import os
import datetime as dt
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from pymongo import MongoClient

load_dotenv()

#Postgres schema helper
PG_SCHEMA = os.getenv("PG_SCHEMA", "public")   # CHANGE: "public" to your own schema name
def qualify(sql: str) -> str:
    # Replace occurrences of {S}.<table> with <schema>.<table>
    return sql.replace("{S}.", f"{PG_SCHEMA}.")

# CONFIG: Postgres and Mongo Queries
CONFIG = {
    "postgres": {
        "enabled": True,
        "uri": os.getenv("PG_URI", "postgresql+psycopg2://postgres:password@localhost:5432/postgres"),  # Will read from your .env file
        "queries": {
            #CHANGE: Replace all the following Postgres queries with your own queries, for each user you identified for your project's Information System
            # Each query must have a unique name, an SQL string, a chart specification, tags (for user roles), and optional params (parameters)
            # :doctor_id, :nurse_id, :patient_name, etc., are placeholders. Their values will come from the dashboard sidebar.
            #User 1: Administrator
            "Admin: list all users (table)": {
                "sql": """
                    SELECT user_id,name, email, phone_number, role
                    FROM {S}.users
                    WHERE role = 'Admin';
                """,
                "chart": {"type": "table"},
                "tags": ["admin"],
                "params": ["user_id"]
            },
            "Admin: devices per room (bar)": {  
                "sql": """
                    SELECT r.room_name, COUNT(*) AS device_count
                    FROM {S}.devices d
                    JOIN {S}.rooms r ON d.room_id = r.room_id
                    GROUP BY r.name
                    ORDER BY device_count DESC;
                """,
                "chart": {"type": "bar", "x": "room_name", "y": "device_count"},
                "tags": ["admin"],
                "params": ["device_id"]
            },

            #User 2: Family member 
            "Family: my devices (table)": {
                "sql": """
                    SELECT d.device_id, d.name, d.device_type, d.firmware_version
                    FROM {S}.devices d
                    JOIN {S}.rules r ON r.device_id = d.device_id
                    JOIN {S}.users u ON u.user_id = r.user_id
                    WHERE u.name = 'Zhang San'
                    ORDER BY d.name;
                """,
                "chart": {"type": "table"},
                "tags": ["user"],
                "params": ["device_id"]
            },
      
            #User 3: Guest 
            "Guest: public-area devices": {
                "sql": """
                    SELECT d.device_id, d.name AS device_name, d.device_type, r.name AS room_name
                    FROM {S}.devices d
                    JOIN {S}.rooms r ON d.room_id = r.room_id
                    WHERE r.name IN ('Living Room','Kitchen')
                    ORDER BY r.name, d.name;
                """,
                "chart": {"type": "table"},
                "tags": ["room"],
                "params": ["device_id"]
            },
            "Guest: device owner lookup": {
                "sql": """
                    SELECT DISTINCT
                            d.device_id,
                            d.name        AS device_name,
                            d.device_type,
                            rm.name       AS room_name,
                            u.name        AS owner_name
                    FROM {S}.devices d
                    JOIN {S}.rules  r  ON r.device_id = d.device_id
                    JOIN {S}.rooms  rm ON rm.room_id  = d.room_id
                    JOIN {S}.users  u  ON u.user_id   = r.user_id
                    WHERE rm.name IN ('Living Room','Kitchen')
                    ORDER BY rm.name, d.name;
                """,
                "chart": {"type": "table"},
                "tags": ["room", "user"],
                "params": ["device_id"]
            }            
        }
    },

    "mongo": {
        "enabled": True,
        "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),  # Will read from the .env file
        "db_name": os.getenv("MONGO_DB", "Smart_Home"),               # Will read from the .env file
        
        # CHANGE: Just like above, replace all the following Mongo queries with your own, for the different users you identified
        "queries": {
            "TS: avg temperature per room (24h)": {
                "collection": "temperature",
                "aggregate": [
                    {"$match": {"ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=24)}}},
                    {"$group": {"_id": "$room", "avg_temp": {"$avg": "$temperature_c"}}},
                    {"$sort": {"_id": 1}}
                ],
                "chart": {"type": "line", "x": "_id", "y": "avg_temp"}
            },
            "TS: brightness trend last 24h (family room)": {
                "collection": "brightness",
                "aggregate": [
                    {"$match": {"room": "Bedroom",
                                "ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=24)}}},
                    {"$project": {"hour": {"$hour": "$ts"}, "lux": "$brightness_lux"}},
                    {"$group": {"_id": "$hour", "avgLux": {"$avg": "$lux"}}},
                    {"$sort": {"_id": 1}}
                ],
                "chart": {"type": "line", "x": "_id", "y": "avgLux"}
            },
           
            "Telemetry: motion events per room (30d)": {
                "collection": "motion_events",
                "aggregate": [
                    {"$match": {"ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=30)}}},
                    {"$group": {"_id": "$room", "events": {"$count": {}}}},
                    {"$sort": {"events": -1}}
                ],
                "chart": {"type": "bar", "x": "_id", "y": "events"}
            },
            "Telemetry: latest device status": {
                "collection": "device_status",
                "aggregate": [
                    {"$sort": {"ts": -1}},
                    {"$group": {"_id": "$device_id", "doc": {"$first": "$$ROOT"}}},
                    {"$replaceRoot": {"newRoot": "$doc"}},
                    {"$project": {"_id": 0, "device_id": 1, "status": 1, "battery_percent": 1, "ts": 1}}
                ],
                "chart": {"type": "table"}
            },
            "Telemetry: low-battery wireless sensors": {
                "collection": "device_status",
                "aggregate": [
                    {"$match": {"battery_percent": {"$lt": 20}, "connectivity": "wireless"}},
                    {"$project": {"_id": 0, "device_id": 1, "battery_percent": 1}}
                ],
                "chart": {"type": "table"}
            },
            "Telemetry: triggered video clips last 24h": {
                "collection": "camera",
                "aggregate": [
                    {"$match": {"ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=24)}}},
                    {"$project": {"_id": 0, "camera_id": 1, "room": 1, "ts": 1, "duration_sec": 1, "file_size_mb": 1}}
                ],
                "chart": {"type": "table"}
            },
            "Telemetry: rule execution log last 7d": {
                "collection": "rules_execution",
                "aggregate": [
                    {"$match": {"executed_at": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=7)}}},
                    {"$group": {"_id": "$status", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}}
                ],
                "chart": {"type": "pie", "names": "_id", "values": "count"}
            }
        }
    }
}
# The following block of code will create a simple Streamlit dashboard page
st.set_page_config(page_title="Smart_Home DB Dashboard", layout="wide")
st.title("Smart_Home | Mini Dashboard (Postgres + MongoDB)")

def metric_row(metrics: dict):
    cols = st.columns(len(metrics))
    for (k, v), c in zip(metrics.items(), cols):
        c.metric(k, v)

@st.cache_resource
def get_pg_engine(uri: str):
    return create_engine(uri, pool_pre_ping=True, future=True)

@st.cache_data(ttl=60)
def run_pg_query(_engine, sql: str, params: dict | None = None):
    with _engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

@st.cache_resource
def get_mongo_client(uri: str):
    return MongoClient(uri)

def mongo_overview(client: MongoClient, db_name: str):
    info = client.server_info()
    db = client[db_name]
    colls = db.list_collection_names()
    stats = db.command("dbstats")
    total_docs = sum(db[c].estimated_document_count() for c in colls) if colls else 0
    return {
        "DB": db_name,
        "Collections": f"{len(colls):,}",
        "Total docs (est.)": f"{total_docs:,}",
        "Storage": f"{round(stats.get('storageSize',0)/1024/1024,1)} MB",
        "Version": info.get("version", "unknown")
    }

@st.cache_data(ttl=60)
def run_mongo_aggregate(_client, db_name: str, coll: str, stages: list):
    db = _client[db_name]
    docs = list(db[coll].aggregate(stages, allowDiskUse=True))
    return pd.json_normalize(docs) if docs else pd.DataFrame()

def render_chart(df: pd.DataFrame, spec: dict):
    if df.empty:
        st.info("No rows.")
        return
    ctype = spec.get("type", "table")
    # light datetime parsing for x axes
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass

    if ctype == "table":
        st.dataframe(df, use_container_width=True)
    elif ctype == "line":
        st.plotly_chart(px.line(df, x=spec["x"], y=spec["y"]), use_container_width=True)
    elif ctype == "bar":
        st.plotly_chart(px.bar(df, x=spec["x"], y=spec["y"]), use_container_width=True)
    elif ctype == "pie":
        st.plotly_chart(px.pie(df, names=spec["names"], values=spec["values"]), use_container_width=True)
    elif ctype == "heatmap":
        pivot = pd.pivot_table(df, index=spec["rows"], columns=spec["cols"], values=spec["values"], aggfunc="mean")
        st.plotly_chart(px.imshow(pivot, aspect="auto", origin="upper",
                                  labels=dict(x=spec["cols"], y=spec["rows"], color=spec["values"])),
                        use_container_width=True)
    elif ctype == "treemap":
        st.plotly_chart(px.treemap(df, path=spec["path"], values=spec["values"]), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

# The following block of code is for the dashboard sidebar, where you can pick your users, provide parameters, etc.
with st.sidebar:
    st.header("Connections")
    # These fields are pre-filled from .env file
    pg_uri = st.text_input("Postgres URI", CONFIG["postgres"]["uri"])     
    mongo_uri = st.text_input("Mongo URI", CONFIG["mongo"]["uri"])        
    mongo_db = st.text_input("Mongo DB name", CONFIG["mongo"]["db_name"]) 
    st.divider()
    auto_run = st.checkbox("Auto-run on selection change", value=False, key="auto_run_global")

    st.header("Role & Parameters")
    # CHANGE: Change the different roles, the specific attributes, parameters used, etc., to match your own Information System
    role = st.selectbox("User role", ["Administrator","Family member","guest","all"], index=3)
    user_id = st.number_input("user_id", min_value=1, value=1, step=1)
    device_id = st.number_input("device_id", min_value=1, value=1, step=1)
    
    PARAMS_CTX = {
        "user_id": int(user_id),
        "device_id": int(device_id),
    }

#Postgres part of the dashboard
st.subheader("Postgres")
try:
    
    eng = get_pg_engine(pg_uri)

    with st.expander("Run Postgres query", expanded=True):
        # The following will filter queries by role
        def filter_queries_by_role(qdict: dict, role: str) -> dict:
            def ok(tags):
                t = [s.lower() for s in (tags or ["all"])]
                return "all" in t or role.lower() in t
            return {name: q for name, q in qdict.items() if ok(q.get("tags"))}

        pg_all = CONFIG["postgres"]["queries"]
        pg_q = filter_queries_by_role(pg_all, role)

        names = list(pg_q.keys()) or ["(no queries for this role)"]
        sel = st.selectbox("Choose a saved query", names, key="pg_sel")

        if sel in pg_q:
            q = pg_q[sel]
            sql = qualify(q["sql"])   
            st.code(sql, language="sql")

            run  = auto_run or st.button("▶ Run Postgres", key="pg_run")
            if run:
                wanted = q.get("params", [])
                params = {k: PARAMS_CTX[k] for k in wanted}
                df = run_pg_query(eng, sql, params=params)
                render_chart(df, q["chart"])
        else:
            st.info("No Postgres queries tagged for this role.")
except Exception as e:
    st.error(f"Postgres error: {e}")

# Mongo panel
if CONFIG["mongo"]["enabled"]:
    st.subheader("🍃 MongoDB")
    try:
        mongo_client = get_mongo_client(mongo_uri)   
        metric_row(mongo_overview(mongo_client, mongo_db))

        with st.expander("Run Mongo aggregation", expanded=True):
            mongo_query_names = list(CONFIG["mongo"]["queries"].keys())
            selm = st.selectbox("Choose a saved aggregation", mongo_query_names, key="mongo_sel")
            q = CONFIG["mongo"]["queries"][selm]
            st.write(f"**Collection:** `{q['collection']}`")
            st.code(str(q["aggregate"]), language="python")
            runm = auto_run or st.button("▶ Run Mongo", key="mongo_run")
            if runm:
                dfm = run_mongo_aggregate(mongo_client, mongo_db, q["collection"], q["aggregate"])
                render_chart(dfm, q["chart"])
    except Exception as e:
        st.error(f"Mongo error: {e}")
