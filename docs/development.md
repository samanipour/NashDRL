# Development Guide

## Install

```bash
python -m venv .venv
# PowerShell
.venv\\Scripts\\Activate.ps1
# Linux/macOS
# source .venv/bin/activate
pip install -e .
```

The project pins TraCI and sumolib to `1.27.1` and requires salabim for the optional replay visualizer. SUMO itself must also be installed separately.

## SUMO environment

On Windows, either add the SUMO `bin` directory to `PATH` or set:

```powershell
$env:SUMO_HOME = "C:\\Program Files (x86)\\Eclipse\\Sumo"
```

Verify:

```powershell
sumo --version
netconvert --version
```

The expected target for this repository is SUMO 1.27.1.

## Run the demo

```bash
python scripts/simulate.py --config configs/experiments/small.yaml --mode mock --visualization false --report true
```

For live GUI:

```bash
python scripts/simulate.py --config configs/experiments/small.yaml --mode mock --visualization true --report true
```

## Real data

Set `simulation.mode=real` and `simulation.dataset_path` to a Nash-DRL dataset JSON with the schema produced by `MockDatasetGenerator`.

## Testing

Tests that do not require a local SUMO installation can run everywhere:

```bash
PYTHONPATH=src python -m pytest -q
```

SUMO/GUI integration should be tested on the development machine where SUMO 1.27.1 and desktop GUI support are installed.
