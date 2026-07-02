import sqlite3
import json

db_path = "data/arivu.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Find all edges pointing to price_return
rows = cursor.execute("""
    SELECT cg.version_id, cg.timestamp, cg.algorithm, ce.source, ce.lag, ce.coeff, ce.p_value 
    FROM causal_edges ce
    JOIN causal_graphs cg ON ce.graph_version_id = cg.version_id
    WHERE ce.target = 'price_return'
    ORDER BY cg.timestamp DESC
""").fetchall()

print(f"Total recent edges to price_return: {len(rows)}")
for r in rows:
    print(f"Graph: {r[0][:8]} | Time: {r[1]} | Alg: {r[2]} | Edge: {r[3]}->(lag={r[4]})->price_return | coeff={r[5]:.4f} | p={r[6]:.4f}")

conn.close()
