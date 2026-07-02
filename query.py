import sqlite3

conn = sqlite3.connect('data/arivu.db')
cursor = conn.cursor()

query1 = """
SELECT
    CASE 
        WHEN (JULIANDAY(o.timestamp_closed) - JULIANDAY(d.timestamp_committed)) * 86400 < 60 
        THEN 'garbage (<60s)'
        ELSE 'clean (>=60s)'
    END AS data_quality,
    COUNT(*) as count
FROM decision_objects d
JOIN outcome_records o ON d.id = o.decision_object_id
GROUP BY data_quality;
"""

print("--- Data Quality Breakdown ---")
for row in cursor.execute(query1).fetchall():
    print(f"{row[0]}: {row[1]} rows")

query2 = """
SELECT MAX((JULIANDAY(o.timestamp_closed) - JULIANDAY(d.timestamp_committed)) * 86400) 
FROM decision_objects d
JOIN outcome_records o ON d.id = o.decision_object_id;
"""
max_dur_row = cursor.execute(query2).fetchone()
max_dur = max_dur_row[0] if max_dur_row and max_dur_row[0] is not None else 0.0
print(f"\nMax Cycle Duration: {max_dur:.2f} seconds")

conn.close()
