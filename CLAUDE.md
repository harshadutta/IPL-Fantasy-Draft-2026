# CLAUDE.md — IPL Fantasy Draft 2026

## Project Overview

Data-driven IPL 2026 fantasy cricket draft preparation tool. Build a cheat sheet with expected fantasy points per player using historical data and ML predictions.

## Goal

Win the draft with friends by having the best data-backed player rankings.

## Tech Stack

- Python (Jupyter notebooks or scripts)
- pandas, scikit-learn, XGBoost/LightGBM
- openpyxl for Excel output

## Key Files

- `TODO.md` — Full task breakdown and phased plan
- `data/` — Raw and processed datasets (Kaggle, Cricsheet)
- `notebooks/` — Analysis and modelling
- `output/` — Final draft cheat sheets (Excel/CSV)

## Data Sources

- Kaggle: Dream11 fantasy points dataset, IPL 2008-2025 dataset
- Cricsheet.org: Ball-by-ball data
- CricAPI: Free player stats API
- Dream11 scoring rules: dream11.com/fantasy-cricket/point-system

## Working Notes

- Draft is imminent — prioritise getting a usable cheat sheet over model perfection
- Dream11 scoring: Runs=1pt, Wicket=25pt, Catch=8pt, Captain=2x, VC=1.5x
- No official Dream11 API — use Kaggle pre-processed data instead
