import json
import numpy as np
import pandas as pd
from pathlib import Path

root = Path("MBPO_workspace/evaluation/scientific_v52_fresh")
episodes = pd.read_csv(root / "episodes.csv")

plan = json.loads(
    (root / "scenario_selection.json").read_text()
)

weights = {
    row["selected_scenario_id"]: row["population_days"]
    for row in plan["strata"]
}

data = episodes.pivot(
    index=["scenario_id", "seed"],
    columns="method",
    values="net_energy_kwh"
)

required = ["mbpo", "sac", "fixed_no_motion"]
data = data[required]

assert not data.isna().any().any(), "Missing controller values"
assert data.index.is_unique, "Duplicate scenario/seed pairs"
assert set(data.index.get_level_values("scenario_id")) == set(weights), "Incomplete sampling coverage"

w = np.array([
    weights[sid]
    for sid in data.index.get_level_values("scenario_id")
], dtype=float)

for method in ("sac", "mbpo"):
    diff = data[method] - data["fixed_no_motion"]

    print("\n", method.upper())
    print("Weighted mean gain (kWh):",
          np.average(diff, weights=w))
    print("Sampled days beating fixed:",
          int((diff > 0).sum()), "/", len(diff))
    print("Weighted fraction beating fixed:",
          np.average((diff > 0).astype(float), weights=w))
    print("Worst five paired days:")
    print(diff.nsmallest(5).to_string())

print("\n2018-01-02 evaluation:")
print(data.loc["2018-01-02"].to_string())