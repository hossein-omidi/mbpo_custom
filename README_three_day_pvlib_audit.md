# Stage 1 — NSRDB → pvlib → PVWatts/SAPM audit

Run the attached Python script from the **root directory** of your own `PVwatt_dev` checkout. The script does not modify training code, GitHub, data, or model weights.

## Requirements

- Original repository with `SAC_workspace`, `mbpo/env`, and `examples/config` directories
- Your `data/pv_weather/nsrdb/newyork_multiyear_manifest.json` and referenced raw 5-minute NSRDB year CSV files
- Python dependencies used by your environment, including pvlib, pandas, numpy, matplotlib, gym, PyTorch as required by the repository

## Usage

Copy `run_three_day_pvlib_audit.py` to the repository root, then run:

```bash
python run_three_day_pvlib_audit.py --start-date 2019-12-15 --days 3 --outdir outputs/pvlib_three_day_audit
```

`--start-date` is the first calendar scenario date; the next two must also exist in the manifest. The full three-day weather timeline is **three consecutive configured 16h15 episodes**, each from 08:45 UTC to 01:00 UTC on the next date. No invented weather is used.

## Outputs

- `three_day_irradiance.png`: NSRDB DNI/DHI/GHI and the fixed-panel POA irradiance.
- `three_day_dc_power.png`: actual fixed 30° south PVWatts/SAPM DC power and an instantaneous sun-normal benchmark (not slew-limited, no motor costs).
- `three_day_temperature_wind.png`: ambient temperature, computed cell temperature, and wind.
- `three_day_pvlib_audit.csv`: complete per-timestamp data and diagnostics.
- `three_day_audit_summary.json`: per-day energy integration, irradiance-closure diagnostics, and independent pvlib parity error.

## Scientific limitations

1. **Parities to direct pvlib calls are implementation checks**, not validation against measured power.
2. NSRDB radiation is modeled (satellite/reanalysis) information, not local ground measurements.
3. The instantaneous sun-normal reference is not a realizable actuator trajectory, does not subtract movement energy, and is not necessarily a POA maximum under diffuse illumination.
4. Episode `reward` sums right-endpoint DC power at 5-minute steps; numerical integration sensitivity to trapezoidal approximation is included.
5. The script checks the audited branch contract (196 timestamps and 195 steps). If your local patched configuration changes these, revise the contract explicitly rather than silently comparing incompatible environments.
6. The report requires original annual NSRDB CSVs, not only a manifest.

## Local validation performed in this environment

- Python syntax compile: PASS
- Command-line help: PASS
- Plot/summarization smoke test using **synthetic fixtures only**: PASS
- Full pvlib + actual NSRDB runtime: NOT RUN (pvlib and annual NSRDB CSVs unavailable here)
