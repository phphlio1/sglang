"""Regenerate phase2_tables_manifest.json from the phase2_policies tree.

Groups run dirs by (seed, arm label); quarantined dirs live outside the tree
(phase2_policies_quarantine/), so they are skipped by construction. Arm
derivation mirrors the table labels: boost+adaptive -> "boost ada j%g",
boost -> "boost g%g", boost-cache+adaptive -> "bc ada j%g",
boost-cache -> "bc l%g m%d", else the policy name.

    python3 analysis/regen_phase2_manifest.py
"""

import json
from pathlib import Path

E2_ROOT = Path(__file__).resolve().parents[1]
TREE = E2_ROOT / "results/glm53/agentx_dynamo/policies_v2/phase2_policies"
OUT = Path(__file__).resolve().parent / "phase2_tables_manifest.json"


def arm_label(m: dict) -> str:
    if m["policy"] == "boost" and m.get("boost_adaptive_gamma"):
        return "boost ada j%g" % m["boost_adaptive_max_jump_s"]
    if m["policy"] == "boost":
        return "boost g%g" % m["boost_gamma"]
    if m["policy"] == "boost-cache" and m.get("boost_adaptive_gamma"):
        return "bc ada j%g" % m["boost_adaptive_max_jump_s"]
    if m["policy"] == "boost-cache":
        return "bc l%g m%d" % (m["boost_cache_lambda"], m["boost_cache_cap_tokens"])
    return m["policy"]


def main() -> None:
    man = {}
    for d in sorted(TREE.glob("*/*/")):
        if not (d / "e2_metrics.json").exists() or not (d / "run_meta.json").exists():
            continue
        m = json.loads((d / "run_meta.json").read_text())
        man.setdefault(str(m["seed"]), {}).setdefault(arm_label(m), []).append(
            str(d.relative_to(E2_ROOT))
        )
    OUT.write_text(json.dumps(man, indent=2) + "\n")
    print({s: {a: len(v) for a, v in arms.items()} for s, arms in man.items()})


if __name__ == "__main__":
    main()
