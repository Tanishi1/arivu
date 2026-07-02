import sqlite3
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
db_path = Path(os.getenv("SQLITE_PATH", "data/arivu.db"))

conn = sqlite3.connect(db_path)
try:
    conn.execute("ALTER TABLE decision_objects ADD COLUMN meta_params_used TEXT;")
    conn.commit()
    print("Added meta_params_used column to decision_objects table.")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e):
        print("Column already exists.")
    else:
        print(f"Error: {e}")
finally:
    conn.close()
