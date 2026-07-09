#!/usr/bin/env python3
import os
import sys
import csv
import json
import sqlite3
import re
import datetime
import subprocess
from pathlib import Path
from collections import Counter
import numpy as np

# Configure stdout and stderr to use UTF-8 encoding on Windows to prevent CP1252 charmap crashes
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ANSI colors for beautiful terminal output
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

ARIVU_LOG_PATH = Path("logs/arivu.log")
RL_LOG_PATH = Path("logs/rl_arm.log")
DB_PATH = Path("data/arivu.db")
TRAINING_BUFFER_PATH = Path("data/training_buffer.csv")
ML1_BUFFER_PATH = Path("data/models/ml1_causal_agent_buffer.json")


def print_header(title):
    print(f"\n{C_BOLD}{C_GREEN}=== {title} ==={C_RESET}")


def get_local_to_utc_offset():
    local_now = datetime.datetime.now()
    utc_now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    return local_now - utc_now


def log_time_to_utc_iso(log_time_str):
    # Parse log time (e.g. "2026-07-07 15:21:04")
    try:
        dt = datetime.datetime.strptime(log_time_str[:19], "%Y-%m-%d %H:%M:%S")
        offset = get_local_to_utc_offset()
        utc_dt = dt - offset
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None


def get_running_python_processes():
    processes = []
    if sys.platform == "win32":
        try:
            # Query command line, creation date and PID using wmic
            cmd = 'wmic process where "name=\'python.exe\'" get commandline,creationdate,processid /format:list'
            output = subprocess.check_output(cmd, shell=True, text=True)
            
            # Parse key-value pairs
            proc_dict = {}
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    proc_dict[k.strip()] = v.strip()
                    
                # Once we have all three fields, save it and reset
                if "ProcessId" in proc_dict and "CommandLine" in proc_dict and "CreationDate" in proc_dict:
                    processes.append(proc_dict.copy())
                    proc_dict.clear()
        except Exception:
            pass
    return processes


def parse_wmic_date(date_str):
    # e.g. "20260707152109.372593+330" -> formatted datetime
    if not date_str or len(date_str) < 14:
        return date_str
    try:
        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:8]
        hour = date_str[8:10]
        minute = date_str[10:12]
        second = date_str[12:14]
        return f"{year}-{month}-{day} {hour}:{minute}:{second}"
    except Exception:
        return date_str


def find_latest_restart(log_path, pattern):
    if not log_path.exists():
        return None, 0
    
    latest_start_line = None
    latest_line_idx = -1
    
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if pattern in line:
                latest_start_line = line
                latest_line_idx = idx
                
    return latest_start_line, latest_line_idx


def main():
    print(f"{C_BOLD}{C_CYAN}Arivu System Diagnostic Report{C_RESET}")
    print(f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # -------------------------------------------------------------------------
    # 1. Active Processes Check
    # -------------------------------------------------------------------------
    print_header("Active Python Processes")
    python_procs = get_running_python_processes()
    
    main_found = False
    rl_found = False
    
    for proc in python_procs:
        cmdline = proc.get("CommandLine", "")
        pid = proc.get("ProcessId", "")
        created = parse_wmic_date(proc.get("CreationDate", ""))
        
        status_str = ""
        # Check rl_main.py FIRST because main.py is a substring of it
        if "rl_main.py" in cmdline:
            status_str = f" {C_GREEN}[RL Arms - rl_main.py]{C_RESET}"
            rl_found = True
        elif "main.py" in cmdline:
            status_str = f" {C_GREEN}[CausalAgent - main.py]{C_RESET}"
            main_found = True
            
        print(f"PID: {C_YELLOW}{pid:<8}{C_RESET} | Created: {C_CYAN}{created}{C_RESET} | Cmd: {cmdline}{status_str}")
        
    if not main_found:
        print(f"{C_RED}WARNING: main.py process not found running!{C_RESET}")
    if not rl_found:
        print(f"{C_RED}WARNING: rl_main.py process not found running!{C_RESET}")

    # -------------------------------------------------------------------------
    # 2. Restart Events and Log Boundaries
    # -------------------------------------------------------------------------
    print_header("Restart Events")
    
    # arivu.log
    arivu_start_line, arivu_start_idx = find_latest_restart(ARIVU_LOG_PATH, "Arivu starting up...")
    arivu_restart_time = None
    if arivu_start_line:
        timestamp_match = re.search(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", arivu_start_line)
        if timestamp_match:
            arivu_restart_time = timestamp_match.group(0)
        print(f"Latest {C_CYAN}arivu.log{C_RESET} restart: Line {C_YELLOW}{arivu_start_idx+1}{C_RESET} at {C_CYAN}{arivu_restart_time}{C_RESET}")
    else:
        print(f"{C_YELLOW}No startup event found in arivu.log{C_RESET}")

    # rl_arm.log
    rl_start_line, rl_start_idx = find_latest_restart(RL_LOG_PATH, "RL arms starting up")
    rl_restart_time = None
    if rl_start_line:
        timestamp_match = re.search(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", rl_start_line)
        if timestamp_match:
            rl_restart_time = timestamp_match.group(0)
        print(f"Latest {C_CYAN}rl_arm.log{C_RESET} restart: Line {C_YELLOW}{rl_start_idx+1}{C_RESET} at {C_CYAN}{rl_restart_time}{C_RESET}")
    else:
        print(f"{C_YELLOW}No startup event found in rl_arm.log{C_RESET}")

    # -------------------------------------------------------------------------
    # 3. Database Outcome Analysis (Since Restart)
    # -------------------------------------------------------------------------
    print_header("CausalAgent Database Outcomes (Since Restart)")
    if DB_PATH.exists() and arivu_restart_time:
        utc_restart_iso = log_time_to_utc_iso(arivu_restart_time)
        if utc_restart_iso:
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Total closed CausalAgent trades since restart
                cursor.execute(
                    "SELECT COUNT(*) FROM outcome_records JOIN decision_objects "
                    "ON outcome_records.decision_object_id = decision_objects.id "
                    "WHERE decision_objects.strategy_name = 'CausalAgent' "
                    "AND decision_objects.timestamp_committed >= ?", (utc_restart_iso,)
                )
                total_recent = cursor.fetchone()[0]
                
                # Escape Valve vs Natural breakdown since restart
                cursor.execute(
                    "SELECT json_extract(decision_objects.meta_params, '$.is_escape_valve'), COUNT(*) "
                    "FROM outcome_records JOIN decision_objects "
                    "ON outcome_records.decision_object_id = decision_objects.id "
                    "WHERE decision_objects.strategy_name = 'CausalAgent' "
                    "AND decision_objects.timestamp_committed >= ? "
                    "GROUP BY 1", (utc_restart_iso,)
                )
                breakdown = cursor.fetchall()
                
                ev_count = 0
                natural_count = 0
                for is_ev, count in breakdown:
                    if is_ev == 1 or is_ev == "1":
                        ev_count = count
                    else:
                        natural_count = count
                        
                print(f"Total Trades Closed: {C_CYAN}{total_recent}{C_RESET}")
                print(f"  └─ Escape Valve Trades: {C_YELLOW}{ev_count}{C_RESET}")
                print(f"  └─ Natural Trades:      {C_GREEN}{natural_count}{C_RESET}")
                
                # Regime Distribution since restart
                cursor.execute(
                    "SELECT json_extract(decision_objects.meta_params, '$.regime'), COUNT(*) "
                    "FROM outcome_records JOIN decision_objects "
                    "ON outcome_records.decision_object_id = decision_objects.id "
                    "WHERE decision_objects.strategy_name = 'CausalAgent' "
                    "AND decision_objects.timestamp_committed >= ? "
                    "GROUP BY 1", (utc_restart_iso,)
                )
                regimes = cursor.fetchall()
                if regimes and total_recent > 0:
                    print("  Regime Distribution:")
                    for r_name, r_cnt in sorted(regimes, key=lambda x: x[1], reverse=True):
                        r_name = r_name or "unknown"
                        pct = (r_cnt / total_recent) * 100
                        print(f"    └─ {r_name:<12}: {C_CYAN}{r_cnt:<3}{C_RESET} ({pct:.1f}%)")
                
                # List recent trades
                print(f"\n{C_BOLD}Recent 15 CausalAgent Trades:{C_RESET}")
                cursor.execute(
                    "SELECT decision_objects.id, decision_objects.timestamp_committed, "
                    "outcome_records.actual_pnl, outcome_records.breach_timestamps, "
                    "decision_objects.tuned_params, decision_objects.meta_params "
                    "FROM decision_objects JOIN outcome_records "
                    "ON decision_objects.id = outcome_records.decision_object_id "
                    "WHERE decision_objects.strategy_name = 'CausalAgent' "
                    "AND decision_objects.timestamp_committed >= ? "
                    "ORDER BY decision_objects.timestamp_committed DESC LIMIT 15", (utc_restart_iso,)
                )
                recent_trades = cursor.fetchall()
                
                if recent_trades:
                    print(f"{'Trade ID (Type)':<17} | {'Committed Time (UTC)':<26} | {'P&L % (USD)':<24} | {'Assumptions Status / Breach Details':<40}")
                    print("-" * 122)
                    for tid, tcommitted, pnl, breach_ts, tuned_json, meta_json in recent_trades:
                        tuned = json.loads(tuned_json) if tuned_json else {}
                        meta = json.loads(meta_json) if meta_json else {}
                        
                        # Determine if EV
                        is_ev = bool(meta.get("is_escape_valve", False))
                        type_str = "EV" if is_ev else "NAT"
                        
                        # Compute exposure
                        cap = tuned.get("capital_allocated", 1000000.0)
                        frac = meta.get("position_fraction_adjusted")
                        if frac is None:
                            frac = tuned.get("position_fraction", 0.10)
                        exposure = frac * cap
                        if exposure <= 0.0:
                            exposure = 20.0
                            
                        # Format P&L %
                        ret = (pnl / exposure) * 100 if (pnl is not None and exposure > 0) else None
                        
                        if ret is not None and pnl is not None:
                            pnl_val_str = f"${pnl:+.2f}" if abs(pnl) >= 0.01 else f"${pnl:+.4f}"
                            pnl_str = f"{ret:+.4f}% ({pnl_val_str})"
                            pnl_color = C_GREEN if ret >= 0 else C_RED
                            pnl_str = f"{pnl_color}{pnl_str:<22}{C_RESET}"
                        else:
                            pnl_str = f"{'N/A':<22}"
                            
                        # Format breach details
                        status_str = f"{C_GREEN}HELD (No Breaches){C_RESET}"
                        if breach_ts:
                            try:
                                breaches = json.loads(breach_ts)
                                if breaches:
                                    details = []
                                    for edge, info in breaches.items():
                                        reason = info.get("reason", "unknown")
                                        # shorten edge name for display
                                        short_edge = edge.replace("causal_edge|", "")
                                        details.append(f"{short_edge} ({C_RED}{reason}{C_RESET})")
                                    status_str = f"{C_YELLOW}BREACHED: {', '.join(details)}{C_RESET}"
                            except Exception:
                                status_str = f"{C_RED}Failed to parse breach details{C_RESET}"
                                
                        print(f"{f'{tid[:8]} ({type_str})':<17} | {tcommitted:<26} | {pnl_str} | {status_str}")
                else:
                    print("No CausalAgent trades closed since the restart.")
                    
                
                # Database Overview (Counts)
                print(f"\n{C_BOLD}Database Table Counts:{C_RESET}")
                for tbl in ["decision_objects", "outcome_records", "layer1_runs", "layer2_trust_scores", "meta_optimizer_state", "causal_graphs", "causal_edges"]:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
                        print(f"  └─ {tbl:<25}: {C_CYAN}{cursor.fetchone()[0]}{C_RESET}")
                    except Exception:
                        pass
                
                # Causal Edge stability
                print(f"\n{C_BOLD}Top 5 Most Stable Causal Edges (Discovery):{C_RESET}")
                try:
                    cursor.execute("""
                        SELECT edge_key, SUM(present)*1.0/COUNT(*) as stability, COUNT(*) as runs
                        FROM layer1_runs GROUP BY edge_key
                        ORDER BY stability DESC LIMIT 5
                    """)
                    stable = cursor.fetchall()
                    for edge, stab, runs in stable:
                        print(f"  └─ {edge:<45} | stability={C_GREEN}{stab:.3f}{C_RESET} (runs={runs})")
                except Exception as e:
                    print(f"  No stability metrics available: {e}")

                conn.close()
            except Exception as e:
                print(f"{C_RED}Database query failed: {e}{C_RESET}")
        else:
            print(f"{C_RED}Could not parse UTC conversion for restart time.{C_RESET}")
    else:
        print(f"{C_YELLOW}SQLite database not found or no restart timestamp available.{C_RESET}")

    print_header("Training Buffer & ML2 Status")
    clean_count = 0
    total_rows = 0
    ev_rows = 0
    breach_count = 0
    if TRAINING_BUFFER_PATH.exists():
        try:
            with open(TRAINING_BUFFER_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_rows += 1
                    is_ev = row.get("is_escape_valve") == "1"
                    if is_ev:
                        ev_rows += 1
                    else:
                        clean_count += 1
                        if row.get("breached") == "1":
                            breach_count += 1
            breach_rate = breach_count / clean_count * 100 if clean_count > 0 else 0
            print(f"Total Rows in Buffer:       {C_CYAN}{total_rows}{C_RESET}  (EV={ev_rows} | NAT={clean_count})")
            print(f"Clean (Non-EV) Samples:     {C_GREEN}{clean_count}{C_RESET} / 30 required for ML2 training  |  Breach rate: {C_YELLOW}{breach_rate:.1f}%{C_RESET}")
            if clean_count < 30:
                print(f"ML2 status:                 {C_YELLOW}Bootstrap Mode{C_RESET} (needs {30 - clean_count} more clean samples)")
            else:
                print(f"ML2 status:                 {C_GREEN}Trained Mode Active{C_RESET}")
        except Exception as e:
            print(f"{C_RED}Failed to read training buffer CSV: {e}{C_RESET}")
    else:
        print(f"{C_YELLOW}training_buffer.csv not found.{C_RESET}")

    # ML2 Checkpoint History from DB
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT checkpoint_number, phase, ml2_brier_score, training_sample_count, timestamp
                FROM model_checkpoints ORDER BY checkpoint_number ASC
            """)
            checkpoints = cursor.fetchall()
            conn.close()
            if checkpoints:
                print(f"\n{C_BOLD}ML2 Checkpoint History:{C_RESET}")
                print(f"  {'#':<4} {'Phase':<12} {'Brier':<10} {'Samples':<10} Timestamp")
                print(f"  {'-'*60}")
                prev_brier = None
                for ckpt_num, phase, brier, samples, ts in checkpoints:
                    if prev_brier is not None and brier < prev_brier:
                        trend = f"{C_GREEN}▼ better{C_RESET}"
                    elif prev_brier is not None and brier > prev_brier:
                        trend = f"{C_RED}▲ worse{C_RESET}"
                    else:
                        trend = f"{C_CYAN}first{C_RESET}"
                    brier_color = C_GREEN if brier < 0.30 else (C_YELLOW if brier < 0.35 else C_RED)
                    print(f"  #{ckpt_num:<3} {phase:<12} {brier_color}{brier:.4f}{C_RESET}     {samples:<10} {ts[:19]}  {trend}")
                    prev_brier = brier
        except Exception as e:
            print(f"{C_RED}  Failed to read ML2 checkpoints: {e}{C_RESET}")

    # -------------------------------------------------------------------------
    # 5. ML1 Status
    # -------------------------------------------------------------------------
    print_header("ML1 Status")
    if ML1_BUFFER_PATH.exists():
        try:
            samples = json.loads(ML1_BUFFER_PATH.read_text(encoding="utf-8"))
            print(f"In-memory sample buffer:    {C_CYAN}{len(samples)}{C_RESET} / 100 needed for retraining")
        except Exception as e:
            print(f"{C_RED}Failed to read ML1 buffer json: {e}{C_RESET}")
    else:
        print(f"In-memory sample buffer:    {C_GREEN}0{C_RESET} / 100 (Model is trained and buffer is cleared)")

    # -------------------------------------------------------------------------
    # 6. Layer 2 Trust Scores
    # -------------------------------------------------------------------------
    print_header("Layer 2 Trust Scores")
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT edge_key, trust, n_observations, successes, failures
                FROM layer2_trust_scores
                ORDER BY n_observations DESC
            """)
            l2_rows = cursor.fetchall()
            conn.close()
            if l2_rows:
                print(f"  {'Edge':<45} {'Trust':>7}  {'N':>4}  {'Win':>4}  {'Loss':>4}  Rating")
                print(f"  {'-'*75}")
                for edge_key, trust, n_obs, successes, failures in l2_rows:
                    if trust >= 0.65:
                        rating = f"{C_GREEN}TRUSTED{C_RESET}"
                    elif trust >= 0.50:
                        rating = f"{C_CYAN}NEUTRAL{C_RESET}"
                    elif trust >= 0.35:
                        rating = f"{C_YELLOW}WEAK{C_RESET}"
                    else:
                        rating = f"{C_RED}DISTRUST{C_RESET}"
                    trust_color = C_GREEN if trust >= 0.65 else (C_YELLOW if trust >= 0.50 else C_RED)
                    print(f"  {edge_key:<45} {trust_color}{trust:.4f}{C_RESET}  {n_obs:>4}  {successes:>4}  {failures:>4}  {rating}")
                # Summary
                trusted = sum(1 for _, t, n, _, _ in l2_rows if t >= 0.65 and n >= 3)
                distrusted = sum(1 for _, t, n, _, _ in l2_rows if t < 0.35 and n >= 3)
                print(f"\n  Trusted edges (≥0.65, N≥3): {C_GREEN}{trusted}{C_RESET}  |  Distrusted (<0.35, N≥3): {C_RED}{distrusted}{C_RESET}")
            else:
                print(f"  {C_YELLOW}No Layer 2 trust scores yet.{C_RESET}")
        except Exception as e:
            print(f"{C_RED}  Layer 2 query failed: {e}{C_RESET}")

    # -------------------------------------------------------------------------
    # 7. MetaOptimizer State per Regime
    # -------------------------------------------------------------------------
    print_header("MetaOptimizer State (per Regime)")
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        from ml.meta_optimizer import MetaParameterOptimizer
        
        opt = MetaParameterOptimizer(db_path=str(DB_PATH))
        
        # Access internal state directly for reporting
        if opt._state:
            for regime_name, bucket in sorted(opt._state.items()):
                if regime_name.lower() not in ("calm", "trending", "volatile"):
                    continue
                population = bucket.get("population", [])
                history = bucket.get("history", [])
                if not population:
                    continue
                
                # Compute actual kernel score for each candidate
                scored_candidates = []
                for cand in population:
                    score = opt._score_config(cand.to_dict(), history)
                    scored_candidates.append((score, cand))
                
                # Pick the best
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                best_score, best = scored_candidates[0]
                
                # For updated_at, grab from DB directly
                updated_at = "?"
                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        row = conn.execute("SELECT updated_at FROM meta_optimizer_state WHERE regime=?", (regime_name,)).fetchone()
                        if row: updated_at = row[0][:19]
                except:
                    pass

                score_color = C_GREEN if best_score >= 0.70 else (C_YELLOW if best_score >= 0.55 else C_RED)
                print(f"  {C_BOLD}[{regime_name.upper()}]{C_RESET}  best_score={score_color}{best_score:.4f}{C_RESET}  "
                      f"threshold={C_CYAN}{best.threshold}{C_RESET}  "
                      f"tau_max={best.tau_max}  "
                      f"pcmci_alpha={best.pcmci_alpha}  "
                      f"k_runs={best.k_runs}  "
                      f"min_runs={best.min_runs}  "
                      f"updated={updated_at}")
        else:
            print(f"  {C_YELLOW}No MetaOptimizer state found.{C_RESET}")
    except Exception as e:
        print(f"{C_RED}  MetaOptimizer query failed: {e}{C_RESET}")


    # -------------------------------------------------------------------------
    # 8. Log Scan Metrics (Since Restart)
    # -------------------------------------------------------------------------
    print_header("Log Metrics (Since Restart)")
    
    # Read relevant lines of arivu.log since startup
    arivu_lines = []
    if ARIVU_LOG_PATH.exists() and arivu_start_idx >= 0:
        with open(ARIVU_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            if arivu_start_idx < len(lines):
                arivu_lines = lines[arivu_start_idx:]
                
    # Read relevant lines of rl_arm.log since startup
    rl_lines = []
    if RL_LOG_PATH.exists() and rl_start_idx >= 0:
        with open(RL_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            if rl_start_idx < len(lines):
                rl_lines = lines[rl_start_idx:]

    # a. Regime retraining checks
    print(f"{C_BOLD}Regime Classifier Retraining Events:{C_RESET}")
    regime_events = [line.strip() for line in arivu_lines if "regime classifier" in line.lower() and "retrained" in line.lower()]
    if regime_events:
        for event in regime_events:
            print(f"  {C_GREEN}[OK]{C_RESET} {event}")
    else:
        print("  No Regime Classifier retraining events yet.")

    # b. ML1 retraining checks
    print(f"\n{C_BOLD}ML1 Retraining Events:{C_RESET}")
    ml1_events = [line.strip() for line in arivu_lines if "ml1 retrained" in line.lower()]
    if ml1_events:
        for event in ml1_events:
            print(f"  {C_GREEN}[OK]{C_RESET} {event}")
    else:
        print("  No ML1 retraining events yet.")

    # Primary Edge Exclusion checks
    print(f"\n{C_BOLD}Primary Edge Exclusions (Since Restart):{C_RESET}")
    rejected_count = sum(1 for line in arivu_lines if "primary edge rejected" in line.lower())
    if rejected_count > 0:
        print(f"  Pruned candidate chains: {C_RED}{rejected_count}{C_RESET}")
    else:
        print("  No primary edges rejected yet.")

    # c. ML2 retraining checks
    print(f"\n{C_BOLD}ML2 Retraining Events:{C_RESET}")
    ml2_events = [line.strip() for line in arivu_lines if "ml2 retrained" in line.lower()]
    if ml2_events:
        for event in ml2_events:
            print(f"  {C_GREEN}[OK]{C_RESET} {event}")
        # Accurate cycle counter from log events
        retrain_count = sum(1 for line in arivu_lines if "ml2 retrained" in line.lower() and "| ml2 |" in line.lower())
        entries_since = 0
        # Count closed trades since last retrain log line
        last_retrain_line_idx = max(
            (i for i, line in enumerate(arivu_lines) if "ml2 retrained" in line.lower() and "| ml2 |" in line.lower()),
            default=-1
        )
        if last_retrain_line_idx >= 0:
            entries_since = sum(
                1 for line in arivu_lines[last_retrain_line_idx:]
                if "writer | ledger closed" in line.lower()
            )
        print(f"  Retrains this session: {C_CYAN}{retrain_count}{C_RESET}  |  Closed trades since last retrain: {C_YELLOW}{entries_since}{C_RESET} / 20")
    else:
        print("  No ML2 retraining events yet.")

    # d. MetaOptimizer Updates (last 15 lines)
    print(f"\n{C_BOLD}Latest MetaOptimizer Updates:{C_RESET}")
    opt_lines = [line.strip() for line in arivu_lines if "metaoptimizer" in line.lower()]
    if opt_lines:
        for line in opt_lines[-15:]:
            print(f"  {C_CYAN}>{C_RESET} {line}")
    else:
        print("  No MetaOptimizer logs yet.")

    # e. ML2 Logs (last 15 lines)
    print(f"\n{C_BOLD}Latest ML2 logs:{C_RESET}")
    ml2_logs = [line.strip() for line in arivu_lines if "ml2" in line.lower() and "retrain" not in line.lower() and "calibration" not in line.lower()]
    if ml2_logs:
        for line in ml2_logs[-15:]:
            print(f"  {C_CYAN}>{C_RESET} {line}")
    else:
        print("  No ML2 logs yet.")

    # f. Hold Reason Breakdown in arivu.log
    print(f"\n{C_BOLD}Hold Reason Breakdown in arivu.log:{C_RESET}")
    hold_reasons = Counter()
    for line in arivu_lines:
        m = re.search(r"reason=(\w+)", line)
        if m:
            hold_reasons[m.group(1)] += 1
    total_h = sum(hold_reasons.values())
    if total_h > 0:
        for r, c in hold_reasons.most_common():
            print(f"  └─ {r:<30}: {C_YELLOW}{c:<4}{C_RESET} ({c/total_h*100:.1f}%)")
    else:
        print("  No hold reason events found in active log.")

    # g. Causal Agent Fill Slippage Stats in arivu.log
    print(f"\n{C_BOLD}Causal Agent Fill Slippage Stats in arivu.log:{C_RESET}")
    ca_slippages = []
    for line in arivu_lines:
        m = re.search(r"slippage=(-?[\d.]+)", line)
        if m:
            ca_slippages.append(float(m.group(1)))
    if ca_slippages:
        arr = np.array(ca_slippages)
        print(f"  └─ Mean Slippage  : {C_CYAN}{arr.mean():.4f}{C_RESET} USD")
        print(f"  └─ Median Slippage: {C_CYAN}{np.median(arr):.4f}{C_RESET} USD")
        print(f"  └─ Worst Slippage : {C_RED}{arr.min():.4f}{C_RESET} USD")
        bad_fills = sum(1 for s in arr if abs(s) > 0.5)
        print(f"  └─ >$0.50 Slippage: {C_YELLOW}{bad_fills}{C_RESET} / {len(arr)} fills ({bad_fills/len(arr)*100:.1f}%)")
    else:
        print("  No slippage metrics found in active log.")

    # h. RL Arm Fill Slippage Stats in rl_arm.log
    print(f"\n{C_BOLD}RL Arm Fill Slippage Stats in rl_arm.log:{C_RESET}")
    slippages = []
    for line in rl_lines:
        m = re.search(r"slippage=(-?[\d.]+)", line)
        if m:
            slippages.append(float(m.group(1)))
    if slippages:
        arr = np.array(slippages)
        print(f"  └─ Mean Slippage  : {C_CYAN}{arr.mean():.4f}{C_RESET} USD")
        print(f"  └─ Median Slippage: {C_CYAN}{np.median(arr):.4f}{C_RESET} USD")
        print(f"  └─ Worst Slippage : {C_RED}{arr.min():.4f}{C_RESET} USD")
        bad_fills = sum(1 for s in arr if abs(s) > 0.5)
        print(f"  └─ >$0.50 Slippage: {C_YELLOW}{bad_fills}{C_RESET} / {len(arr)} fills ({bad_fills/len(arr)*100:.1f}%)")
    else:
        print("  No slippage metrics found in active log.")

    # h. Error scans
    print(f"\n{C_BOLD}Errors in CausalAgent (arivu.log) since restart:{C_RESET}")
    arivu_errors = [line.strip() for line in arivu_lines if "ERROR" in line]
    if arivu_errors:
        for err in arivu_errors:
            print(f"  {C_RED}[ERR] {err}{C_RESET}")
    else:
        print(f"  {C_GREEN}[OK] No errors found{C_RESET}")
        
    print(f"\n{C_BOLD}Errors in RL Arms (rl_arm.log) since restart:{C_RESET}")
    rl_errors = [line.strip() for line in rl_lines if "ERROR" in line]
    if rl_errors:
        for err in rl_errors:
            print(f"  {C_RED}[ERR] {err}{C_RESET}")
    else:
        print(f"  {C_GREEN}[OK] No errors found{C_RESET}")
        
    # -------------------------------------------------------------------------
    # 9. Empirical Analysis & Lagged Chains Integration
    # -------------------------------------------------------------------------
    emp_path = Path("scratch/empirical_analysis.py")
    lagged_path = Path("scratch/find_lagged_feedback_chains.py")

    if emp_path.exists():
        print_header("Empirical Analysis Results")
        try:
            output = subprocess.check_output([sys.executable, str(emp_path)], text=True, errors="ignore")
            print(output.strip())
        except Exception as e:
            print(f"{C_RED}Failed to run empirical_analysis.py: {e}{C_RESET}")

    if lagged_path.exists():
        print_header("Lagged Feedback Chains Report")
        try:
            output = subprocess.check_output([sys.executable, str(lagged_path)], text=True, errors="ignore")
            print(output.strip())
        except Exception as e:
            print(f"{C_RED}Failed to run find_lagged_feedback_chains.py: {e}{C_RESET}")

    print()


if __name__ == "__main__":
    main()
