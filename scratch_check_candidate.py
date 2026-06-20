# scratch_check_candidate.py
import sqlite3
import json

conn = sqlite3.connect("candidates.db")
cur = conn.cursor()
cur.execute("SELECT name, ai_report FROM candidates WHERE ai_report IS NOT NULL AND ai_report != ''")
rows = cur.fetchall()
conn.close()

found = False
for name, ai_report_str in rows:
    try:
        report = json.loads(ai_report_str)
        if "custom_coding_test" in report:
            print(f"Found candidate with recruitment kit: {name}")
            print(list(report.keys()))
            found = True
            break
    except Exception:
        continue

if not found:
    print("No candidates with a recruitment kit found in the database.")
