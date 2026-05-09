"""
Cathedral / Wake Protocol — post-run exporter.

Reads a finished simulation database (SQLite or PostgreSQL) and writes three
analysis-ready CSVs to ``--out_dir``:

    1. ``macro_timeseries.csv``        per-(world_id, tick) macro metrics
    2. ``agents_final_tick<N>.csv``    flattened final state of every agent
    3. ``compliance_report.csv``       per-world Voluntary Submission Index

The script is invoked as a module so it picks up the project's ORM models
without any path tricks::

    python -m scripts.export_results \\
        --db sqlite:///cathedral_100.db \\
        --out_dir ./results_100

Design notes
------------
* The *only* connection point is the DB URL — the script does not import
  any runtime / async / LLM code. It is therefore safe to run on a separate
  machine from the simulation.
* The macro time-series table (``tick_metrics``) is written by the runner
  during the simulation. If it is missing (older DB) the exporter falls back
  to skipping the macro CSV with a clear warning rather than failing the
  whole job — agent and compliance CSVs are still produced.
* The "final tick" cutoff for the agents CSV is taken to be the *maximum*
  ``tick`` observed in ``agent_states`` per world; this is robust to
  shutdowns that left different worlds at different tick numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger("cathedral.export")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="scripts.export_results",
        description="Export Cathedral / Wake Protocol simulation logs to CSV.",
    )
    p.add_argument(
        "--db",
        required=True,
        help="SQLAlchemy DB URL, e.g. sqlite:///cathedral_100.db or "
             "postgresql+psycopg://user:pw@host/cathedral",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write CSVs into (created if missing).",
    )
    p.add_argument(
        "--final_tick",
        type=int,
        default=None,
        help="Override the 'final tick' label used in the agents CSV "
             "filename. Defaults to max(tick) observed in agent_states.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _table_exists(engine: Engine, table: str) -> bool:
    return table in inspect(engine).get_table_names()


def _decode_json_column(value: Any) -> Any:
    """Some drivers (sqlite) return JSON as ``str``; normalise to dict."""
    if isinstance(value, (dict, list)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


# ---------------------------------------------------------------------------
# 1. Macro time series (per-tick metrics)
# ---------------------------------------------------------------------------


_MACRO_COLUMNS = [
    "world_id",
    "tick",
    "timestamp",
    "mean_authenticity_index",
    "mean_spectacle_immersion",
    "mean_cognitive_load",
    "burnout_rate",
    "automation_triggered",
    "emission_deficit",
    "prices",
    "aggregate_time_allocation",
    "extra",
]


def export_macro_timeseries(engine: Engine, out_path: str) -> Optional[pd.DataFrame]:
    """Read ``tick_metrics`` and write it to ``out_path``.

    The JSON blobs (``prices``, ``aggregate_time_allocation``) are kept as
    strings in the CSV so the file remains parseable by ``pandas.read_csv``
    without external schema hints; downstream notebooks can re-parse with
    ``df['prices'].map(json.loads)``.
    """
    if not _table_exists(engine, "tick_metrics"):
        logger.warning(
            "Table 'tick_metrics' is missing — skipping %s. "
            "Re-run the simulation with the current main.py so per-tick "
            "metrics are persisted, or supply a freshly-produced DB.",
            os.path.basename(out_path),
        )
        return None

    sql = (
        "SELECT world_id, tick, timestamp, mean_authenticity_index, "
        "mean_spectacle_immersion, mean_cognitive_load, burnout_rate, "
        "automation_triggered, emission_deficit, prices, "
        "aggregate_time_allocation, extra "
        "FROM tick_metrics "
        "ORDER BY world_id, tick"
    )
    df = pd.read_sql_query(sql, engine)
    if df.empty:
        logger.warning("tick_metrics is empty — writing empty %s.", out_path)
    else:
        # Normalise JSON columns to compact strings for CSV safety.
        for col in ("prices", "aggregate_time_allocation", "extra"):
            df[col] = df[col].map(
                lambda v: json.dumps(_decode_json_column(v), ensure_ascii=False)
            )

    df = df.reindex(columns=_MACRO_COLUMNS)
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote %s — %d rows.", out_path, len(df))
    return df


# ---------------------------------------------------------------------------
# 2. Agents final tick (flattened payload)
# ---------------------------------------------------------------------------


def _flatten_agent_payload(world_id: str, agent_id: str, tick: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Translate one ``AgentState.model_dump()`` into a flat row."""
    cog = payload.get("cognition", {}) or {}
    eco = payload.get("economics", {}) or {}
    ideo = payload.get("ideology", {}) or {}
    ta = (eco.get("time_allocation") or {})

    archetype_hint = (payload.get("core_identity_prompt") or "")[:80].replace("\n", " ")

    return {
        "agent_id": agent_id,
        "world_id": world_id,
        "tick": tick,
        "archetype_hint": archetype_hint,
        # Ideology
        "ideology_economic_axis": ideo.get("economic_axis"),
        "ideology_social_axis": ideo.get("social_axis"),
        "ideology_conformity_index": ideo.get("conformity_index"),
        # Cognition
        "authenticity_index": cog.get("authenticity_index"),
        "spectacle_immersion": cog.get("spectacle_immersion"),
        "cognitive_load": cog.get("cognitive_load"),
        "burnout_threshold": cog.get("burnout_threshold"),
        "dopamine_baseline": cog.get("dopamine_baseline"),
        # Economics
        "cash_balance": eco.get("cash_balance"),
        "ubi_received": eco.get("ubi_received"),
        "survival_cost": eco.get("survival_cost"),
        # Time allocation
        "labor_h": ta.get("labor"),
        "creation_h": ta.get("creation"),
        "spectacle_h": ta.get("spectacle"),
        "rest_h": ta.get("rest"),
    }


_AGENTS_COLUMNS = [
    "agent_id",
    "world_id",
    "tick",
    "archetype_hint",
    "ideology_economic_axis",
    "ideology_social_axis",
    "ideology_conformity_index",
    "authenticity_index",
    "spectacle_immersion",
    "cognitive_load",
    "burnout_threshold",
    "dopamine_baseline",
    "cash_balance",
    "ubi_received",
    "survival_cost",
    "labor_h",
    "creation_h",
    "spectacle_h",
    "rest_h",
]


def export_agents_final(
    engine: Engine,
    out_dir: str,
    *,
    final_tick_override: Optional[int] = None,
) -> tuple[Optional[pd.DataFrame], int]:
    """Write the final-tick agent snapshot CSV. Returns (df, final_tick)."""
    if not _table_exists(engine, "agent_states"):
        logger.error("Table 'agent_states' missing — cannot export agents.")
        return None, -1

    raw = pd.read_sql_query(
        "SELECT agent_id, world_id, tick, payload FROM agent_states",
        engine,
    )
    if raw.empty:
        logger.warning("agent_states is empty — writing empty agents CSV.")
        final_tick = final_tick_override if final_tick_override is not None else 0
        out_path = os.path.join(out_dir, f"agents_final_tick{final_tick}.csv")
        pd.DataFrame(columns=_AGENTS_COLUMNS).to_csv(out_path, index=False)
        return None, final_tick

    final_tick = (
        final_tick_override
        if final_tick_override is not None
        else int(raw["tick"].max())
    )

    rows = []
    for _, r in raw.iterrows():
        payload = _decode_json_column(r["payload"]) or {}
        if not isinstance(payload, dict):
            logger.warning(
                "Skipping agent %s/%s: payload is not a dict (%s).",
                r["world_id"], r["agent_id"], type(payload).__name__,
            )
            continue
        rows.append(
            _flatten_agent_payload(
                world_id=str(r["world_id"]),
                agent_id=str(r["agent_id"]),
                tick=int(r["tick"]),
                payload=payload,
            )
        )

    df = pd.DataFrame(rows, columns=_AGENTS_COLUMNS)
    df = df.sort_values(["world_id", "agent_id"]).reset_index(drop=True)

    out_path = os.path.join(out_dir, f"agents_final_tick{final_tick}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote %s — %d rows (final_tick=%d).", out_path, len(df), final_tick)
    return df, final_tick


# ---------------------------------------------------------------------------
# 3. Compliance / Voluntary Submission Index report
# ---------------------------------------------------------------------------


_COMPLIANCE_COLUMNS = [
    "world_id",
    "n_offers",
    "n_accepted",
    "voluntary_submission_index",
    "mean_authenticity_delta_accepted",
    "mean_debt_before",
    "mean_forgiveness_amount",
]


def export_compliance_report(engine: Engine, out_path: str) -> Optional[pd.DataFrame]:
    """Aggregate the compliance log into per-world summary statistics.

    For each world we compute:

      * ``n_offers``: number of offers presented (one row per offer).
      * ``n_accepted``: subset that returned ``accepted=True``.
      * ``voluntary_submission_index``: ``n_accepted / n_offers``.
      * ``mean_authenticity_delta_accepted``: average
        ``authenticity_after - authenticity_before`` *for accepted offers*.
        This is the per-capita capitulation cost the experiment is built to
        measure; refusals are excluded by construction (their delta is 0
        because the runner does not mutate refusal-state).
      * ``mean_debt_before`` / ``mean_forgiveness_amount``: descriptive
        anchors so a reader can sanity-check that worlds were comparable
        in terms of the offer they faced.
    """
    if not _table_exists(engine, "compliance_decisions"):
        logger.warning(
            "Table 'compliance_decisions' missing — skipping %s. "
            "This is expected if the Social Credit Protocol never fired "
            "(e.g. no agents indebted at the scheduled tick).",
            os.path.basename(out_path),
        )
        empty = pd.DataFrame(columns=_COMPLIANCE_COLUMNS)
        empty.to_csv(out_path, index=False)
        return empty

    df = pd.read_sql_query(
        "SELECT world_id, accepted, debt_before, forgiveness_amount, "
        "authenticity_before, authenticity_after "
        "FROM compliance_decisions",
        engine,
    )
    if df.empty:
        logger.warning("compliance_decisions is empty — writing empty %s.", out_path)
        empty = pd.DataFrame(columns=_COMPLIANCE_COLUMNS)
        empty.to_csv(out_path, index=False)
        return empty

    # SQLite stores bool as 0/1 — coerce both shapes uniformly.
    df["accepted"] = df["accepted"].astype(bool)
    df["authenticity_delta"] = df["authenticity_after"] - df["authenticity_before"]

    grouped = df.groupby("world_id", sort=True)
    rows = []
    for world_id, sub in grouped:
        n_offers = len(sub)
        accepted = sub[sub["accepted"]]
        n_accepted = len(accepted)
        rows.append(
            {
                "world_id": world_id,
                "n_offers": n_offers,
                "n_accepted": n_accepted,
                "voluntary_submission_index": (
                    n_accepted / n_offers if n_offers else 0.0
                ),
                "mean_authenticity_delta_accepted": (
                    float(accepted["authenticity_delta"].mean())
                    if n_accepted
                    else 0.0
                ),
                "mean_debt_before": float(sub["debt_before"].mean()),
                "mean_forgiveness_amount": float(sub["forgiveness_amount"].mean()),
            }
        )

    out = pd.DataFrame(rows, columns=_COMPLIANCE_COLUMNS)
    out.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote %s — %d worlds.", out_path, len(out))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = _parse_args(argv)
    _ensure_dir(args.out_dir)

    engine = create_engine(args.db, future=True)

    # Sanity: confirm DB is reachable before doing anything expensive.
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        logger.exception("Failed to connect to %s", args.db)
        return 2

    macro_path = os.path.join(args.out_dir, "macro_timeseries.csv")
    export_macro_timeseries(engine, macro_path)

    _, final_tick = export_agents_final(
        engine, args.out_dir, final_tick_override=args.final_tick
    )

    compliance_path = os.path.join(args.out_dir, "compliance_report.csv")
    export_compliance_report(engine, compliance_path)

    logger.info("Export complete. Output dir: %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
