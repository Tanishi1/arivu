"""ml/causal_discovery.py  — Task 3

PCMCI causal discovery engine.

Runs on the rolling 200-bar feature matrix from FeatureBarBuilder and outputs
a versioned GraphSnapshot — a directed acyclic graph of statistically validated
causal relationships between the 19 market variables.

SELF-REFERENTIAL LEAKAGE EXCLUSION — do not remove or comment out.
These CausalState fields are NEVER added to the graph input array.
See implementation_plan.md for full rationale.

    EXCLUDED_FROM_GRAPH = {
        "position_size", "capital_deployed", "active_strategy",
        "last_tick_timestamp", "price"
    }

These are already excluded at the FeatureBarBuilder level (core/feature_bar.py).
This module operates only on the 19 safe variables that builder outputs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TAU_MAX: int = 4            # max lag in bars (4 × 10s = 40s lookahead)
ALPHA: float = 0.05         # significance threshold for PCMCI
MIN_BARS_PCMCI: int = 80    # minimum bars for PCMCI (uses Granger below this)
MIN_BARS_GRANGER: int = 40  # minimum for pairwise Granger fallback
GRANGER_MAXLAG: int = 4     # must match TAU_MAX for consistency

# ---------------------------------------------------------------------------
# GraphSnapshot data structure
# ---------------------------------------------------------------------------

class CausalEdge:
    """A single directed causal edge in the discovered graph."""
    __slots__ = ("source", "target", "lag", "coeff", "p_value",
                 "conditioning_set", "graph_version_id")

    def __init__(
        self,
        source: str,
        target: str,
        lag: int,
        coeff: float,
        p_value: float,
        conditioning_set: list[str],
        graph_version_id: str,
    ) -> None:
        self.source = source
        self.target = target
        self.lag = lag
        self.coeff = coeff
        self.p_value = p_value
        self.conditioning_set = conditioning_set
        self.graph_version_id = graph_version_id

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "lag": self.lag,
            "coeff": self.coeff,
            "p_value": self.p_value,
            "conditioning_set": self.conditioning_set,
            "graph_version_id": self.graph_version_id,
        }


class GraphSnapshot:
    """A versioned, immutable snapshot of the discovered causal graph."""

    def __init__(
        self,
        version_id: str,
        timestamp: datetime,
        edges: list[CausalEdge],
        variables: list[str],
        n_bars_used: int,
        algorithm: str,  # "pcmci" or "granger_fallback"
        feature_means: dict[str, float] = None,
        feature_stds: dict[str, float] = None,
    ) -> None:
        self.version_id = version_id
        self.timestamp = timestamp
        self.edges = edges
        self.variables = variables
        self.n_bars_used = n_bars_used
        self.algorithm = algorithm
        self.feature_means = feature_means or {}
        self.feature_stds = feature_stds or {}

    def edges_to_target(self, target: str) -> list[CausalEdge]:
        """Return all edges whose target is `target`."""
        return [e for e in self.edges if e.target == target]

    def edges_from_source(self, source: str) -> list[CausalEdge]:
        return [e for e in self.edges if e.source == source]

    def summary(self) -> str:
        return (
            f"GraphSnapshot v={self.version_id[:8]} "
            f"alg={self.algorithm} edges={len(self.edges)} "
            f"bars={self.n_bars_used} ts={self.timestamp.isoformat()}"
        )


# ---------------------------------------------------------------------------
# Causal Discovery Engine
# ---------------------------------------------------------------------------

class CausalDiscoveryEngine:
    """Runs PCMCI (or Granger fallback) on the feature matrix.

    Usage:
        engine = CausalDiscoveryEngine(db_path="data/arivu.db")
        snapshot = engine.run(matrix, variable_names)
        # snapshot.edges contains all significant causal relationships
    """

    def __init__(self, db_path: str = "data/arivu.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._latest_snapshot: Optional[GraphSnapshot] = None
        self._ensure_schema()

    def run(
        self,
        matrix: np.ndarray,
        variable_names: list[str],
    ) -> GraphSnapshot:
        """Run causal discovery and return a versioned GraphSnapshot.

        Args:
            matrix:         shape (n_bars, n_vars), float64, no NaN/Inf.
            variable_names: column labels matching VARIABLE_NAMES.

        Returns:
            GraphSnapshot with all significant edges.
        """
        n_bars, n_vars = matrix.shape
        version_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc)

        # Sanitise: replace any NaN/Inf with 0 (should not occur, but guard it)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate means and stds for normalization in the Twin Simulator
        means = np.mean(matrix, axis=0)
        stds = np.std(matrix, axis=0)
        feature_means = {variable_names[i]: float(means[i]) for i in range(n_vars)}
        feature_stds = {variable_names[i]: float(stds[i]) for i in range(n_vars)}

        if n_bars >= MIN_BARS_PCMCI:
            edges, algorithm = self._run_pcmci(matrix, variable_names, version_id)
        elif n_bars >= MIN_BARS_GRANGER:
            edges, algorithm = self._run_granger(matrix, variable_names, version_id)
        else:
            logger.warning(
                "CausalDiscovery: only %d bars available (min %d for Granger). "
                "Returning empty graph.",
                n_bars, MIN_BARS_GRANGER,
            )
            edges, algorithm = [], "none"

        snapshot = GraphSnapshot(
            version_id=version_id,
            timestamp=ts,
            edges=edges,
            variables=variable_names,
            n_bars_used=n_bars,
            algorithm=algorithm,
            feature_means=feature_means,
            feature_stds=feature_stds,
        )

        self._persist(snapshot)

        with self._lock:
            self._latest_snapshot = snapshot

        logger.info(
            "CausalDiscovery | %s | significant edges=%d",
            snapshot.summary(), len(edges),
        )
        return snapshot

    def get_latest(self) -> Optional[GraphSnapshot]:
        with self._lock:
            return self._latest_snapshot

    # ------------------------------------------------------------------
    # PCMCI (primary)
    # ------------------------------------------------------------------

    def _run_pcmci(
        self,
        matrix: np.ndarray,
        variable_names: list[str],
        version_id: str,
    ) -> tuple[list[CausalEdge], str]:
        """Run PCMCI with ParCorr conditional independence test."""
        try:
            from tigramite import data_processing as pp
            from tigramite.pcmci import PCMCI
            from tigramite.independence_tests.parcorr import ParCorr

            dataframe = pp.DataFrame(
                matrix,
                datatime=np.arange(len(matrix)),
                var_names=variable_names,
            )
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
            results = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=ALPHA)

            edges = self._parse_pcmci_results(results, variable_names, version_id)
            return edges, "pcmci"

        except ImportError:
            logger.error("tigramite not installed — falling back to Granger")
            return self._run_granger(matrix, variable_names, version_id)
        except Exception as exc:
            logger.error("PCMCI failed: %s — falling back to Granger", exc)
            return self._run_granger(matrix, variable_names, version_id)

    def _parse_pcmci_results(
        self,
        results: dict,
        variable_names: list[str],
        version_id: str,
    ) -> list[CausalEdge]:
        """Extract significant edges from PCMCI output dict."""
        
        edges: list[CausalEdge] = []
        n_vars = len(variable_names)

        p_matrix = results.get("p_matrix")
        val_matrix = results.get("val_matrix")
        if p_matrix is None or val_matrix is None:
            return edges

        FORBIDDEN_TARGETS = {
            "session_sin", "session_cos", "rsi", 
            "algo_health_p_normal", "algo_health_p_stressed", "algo_health_p_degraded"
        }

        for j in range(n_vars):       # target variable index
            target_name = variable_names[j]
            if target_name in FORBIDDEN_TARGETS:
                continue
                
            # Collect all p-values for this target (only valid tests)
            target_candidates = []
            p_values = []
            
            for i in range(n_vars):   # source variable index
                if i == j:
                    continue
                for tau in range(1, TAU_MAX + 1):  # lag (skip self-loops)
                    p_val = float(p_matrix[i, j, tau])
                    if np.isnan(p_val) or p_val >= 1.0:
                        continue
                    coeff = float(val_matrix[i, j, tau])
                    
                    target_candidates.append((i, tau, p_val, coeff))
                    p_values.append(p_val)
            
            if not p_values:
                continue
                
            reject = [p < ALPHA for p in p_values]
                
            # if target_name == "price_return":
            #     logger.info(
            #         "BH_DEBUG price_return | raw_candidates=%d | p_values=%s | reject=%s",
            #         len(target_candidates),
            #         [round(p, 4) for p in p_values],
            #         list(reject) if p_values else []
            #     )
                
            for idx, (i, tau, p_val, coeff) in enumerate(target_candidates):
                if reject[idx] and abs(coeff) > 1e-6:
                    edges.append(CausalEdge(
                        source=variable_names[i],
                        target=target_name,
                        lag=tau,
                        coeff=coeff,
                        p_value=p_val,
                        conditioning_set=[],  # ParCorr conditions implicitly
                        graph_version_id=version_id,
                    ))
        return edges

    # ------------------------------------------------------------------
    # Granger causality fallback (pairwise)
    # ------------------------------------------------------------------

    def _run_granger(
        self,
        matrix: np.ndarray,
        variable_names: list[str],
        version_id: str,
    ) -> tuple[list[CausalEdge], str]:
        """Pairwise Granger causality tests — used when n_bars < MIN_BARS_PCMCI."""
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError:
            logger.error("statsmodels not installed — returning empty graph")
            return [], "none"

        edges: list[CausalEdge] = []
        n_vars = len(variable_names)
        n_bars = matrix.shape[0]
        
        # Stricter alpha threshold for Granger when data is scarce
        granger_alpha = 0.01 if n_bars < 100 else ALPHA

        FORBIDDEN_TARGETS = {
            "session_sin", "session_cos", "rsi", 
            "algo_health_p_normal", "algo_health_p_stressed", "algo_health_p_degraded"
        }

        for j in range(n_vars):
            target = variable_names[j]
            if target in FORBIDDEN_TARGETS:
                continue
            y = matrix[:, j]

            for i in range(n_vars):
                if i == j:
                    continue
                source = variable_names[i]
                x = matrix[:, i]

                try:
                    data = np.column_stack([y, x])
                    result = grangercausalitytests(
                        data, maxlag=GRANGER_MAXLAG, verbose=False
                    )
                    for lag in range(1, GRANGER_MAXLAG + 1):
                        test_stats = result[lag][0]
                        # Use the F-test p-value
                        p_val = test_stats["ssr_ftest"][1]
                        f_stat = test_stats["ssr_ftest"][0]
                        if p_val < granger_alpha:
                            edges.append(CausalEdge(
                                source=source,
                                target=target,
                                lag=lag,
                                coeff=float(f_stat),
                                p_value=float(p_val),
                                conditioning_set=[],  # pairwise — no conditioning
                                graph_version_id=version_id,
                            ))
                except Exception:
                    continue

        return edges, "granger_fallback"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS causal_graphs (
                    version_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    n_bars_used INTEGER NOT NULL,
                    variables TEXT NOT NULL,
                    edges TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS causal_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    graph_version_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    lag INTEGER NOT NULL,
                    coeff REAL NOT NULL,
                    p_value REAL NOT NULL,
                    conditioning_set TEXT NOT NULL,
                    FOREIGN KEY (graph_version_id) REFERENCES causal_graphs(version_id)
                )
            """)
            conn.commit()

    def _persist(self, snapshot: GraphSnapshot) -> None:
        try:
            edges_dicts = [e.to_dict() for e in snapshot.edges]
            with sqlite3.connect(self._db_path) as conn:
                # Store the variables and stats in a unified JSON structure
                variables_json = json.dumps({
                    "names": snapshot.variables,
                    "means": snapshot.feature_means,
                    "stds": snapshot.feature_stds,
                })
                conn.execute(
                    "INSERT INTO causal_graphs VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        snapshot.version_id,
                        snapshot.timestamp.isoformat(),
                        snapshot.algorithm,
                        snapshot.n_bars_used,
                        variables_json,
                        json.dumps(edges_dicts),
                    ),
                )
                for e in snapshot.edges:
                    conn.execute(
                        "INSERT INTO causal_edges "
                        "(graph_version_id, source, target, lag, coeff, p_value, conditioning_set) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            e.graph_version_id,
                            e.source,
                            e.target,
                            e.lag,
                            e.coeff,
                            e.p_value,
                            json.dumps(e.conditioning_set),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.error("CausalDiscovery: failed to persist graph: %s", exc)
