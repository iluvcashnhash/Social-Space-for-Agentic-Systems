"""Export cathedral_sim.db to CSV and JSONL for analysis."""
import csv
import json
import sqlite3

conn = sqlite3.connect("cathedral_sim.db")
rows = conn.execute(
    "SELECT agent_id, world_id, tick, payload FROM agent_states "
    "ORDER BY world_id, agent_id"
).fetchall()

# ── agents.csv ───────────────────────────────────────────────────────────────
with open("agents.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "agent_id", "world_id", "tick", "archetype_hint",
        "authenticity_index", "spectacle_immersion", "cognitive_load",
        "burnout_threshold", "cash_balance", "ubi_received", "survival_cost",
        "labor_h", "creation_h", "spectacle_h", "rest_h",
        "ideology_economic_axis", "ideology_social_axis", "ideology_conformity_index",
    ])
    for agent_id, world_id, tick, payload in rows:
        p = json.loads(payload) if isinstance(payload, str) else payload
        cog  = p["cognition"]
        eco  = p["economics"]
        ideo = p["ideology"]
        ta   = eco["time_allocation"]
        w.writerow([
            agent_id, world_id, tick,
            p["core_identity_prompt"][:60].replace("\n", " "),
            cog["authenticity_index"],
            cog["spectacle_immersion"],
            cog["cognitive_load"],
            cog["burnout_threshold"],
            eco["cash_balance"],
            eco["ubi_received"],
            eco["survival_cost"],
            ta["labor"], ta["creation"], ta["spectacle"], ta["rest"],
            ideo["economic_axis"],
            ideo["social_axis"],
            ideo["conformity_index"],
        ])

print(f"agents.csv  — {len(rows)} rows")

# ── agents_full.jsonl ────────────────────────────────────────────────────────
with open("agents_full.jsonl", "w", encoding="utf-8") as f:
    for agent_id, world_id, tick, payload in rows:
        p = json.loads(payload) if isinstance(payload, str) else payload
        p["_agent_id"] = agent_id
        p["_world_id"] = world_id
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"agents_full.jsonl — {len(rows)} records")
conn.close()
print("Export complete.")
