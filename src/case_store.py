import sqlite3
from datetime import datetime
import pandas as pd

DB_PATH = "data/demo/nuhga_cases.db"

def init_db():
    """Initialize SQLite case-tracking database with default Shahpura pilot cases."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            location_ref TEXT,
            issue_type TEXT,
            source_evidence TEXT,
            status TEXT,
            reviewer_name TEXT,
            decision_reason TEXT,
            updated_at TEXT
        )
    """)
    cur.execute("SELECT COUNT(*) FROM cases")
    if cur.fetchone()[0] == 0:
        seed_cases = [
            ("CASE-SHP-101", "FLAG-SHP-01 (NE Lake Edge)", "Possible water-body encroachment",
             "Sentinel-2 NDWI drop (+0.34 to -0.18) overlapping lake buffer", "Pending",
             "Unassigned", "Awaiting municipal field engineer inspection", "2026-10-25 10:30"),
            ("CASE-SHP-102", "FLAG-SHP-02 (Northern Inlet)", "Possible drainage obstruction",
             "New built-up surface intersecting primary stormwater channel", "Verified",
             "Er. R. Sharma (BMC)", "Field check confirmed temporary debris obstruction; notice issued", "2026-10-26 14:15"),
            ("CASE-SHP-103", "FLAG-SHP-03 (Western Runoff Path)", "Low-confidence finding",
             "Seasonal vegetation clearing near modelled terrain path", "Dismissed",
             "A. Verma (Town Planning)", "Verified as seasonal grass trimming; no structural filling found", "2026-10-26 16:40")
        ]
        cur.executemany("INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?)", seed_cases)
        conn.commit()
    conn.close()

def get_all_cases() -> pd.DataFrame:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM cases ORDER BY case_id ASC", conn)
    conn.close()
    return df

def update_case_status(case_id: str, new_status: str, reviewer: str, reason: str):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    cur.execute("""
        UPDATE cases
        SET status = ?, reviewer_name = ?, decision_reason = ?, updated_at = ?
        WHERE case_id = ?
    """, (new_status, reviewer, reason, now_str, case_id))
    conn.commit()
    conn.close()

def add_new_case(location_ref: str, issue_type: str, source_evidence: str):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM cases")
    count = cur.fetchone()[0] + 101
    case_id = f"CASE-SHP-{count}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    cur.execute("INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
        case_id, location_ref, issue_type, source_evidence, "Pending", "Citizen / Auto-Screening", "Logged for municipal review", now_str
    ))
    conn.commit()
    conn.close()
    return case_id