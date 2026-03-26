# IPL Fantasy Draft 2026 — Task List

**Draft date:** Tomorrow
**Goal:** Data-driven draft preparation with expected fantasy points per player

---

## Phase 1: Data Collection

- [ ] **Get all 10 IPL 2026 squads** — full player lists post-auction
  - Sources: Olympics.com, Wisden, myKhel
  - Output: single CSV/Excel with columns: Player, Team, Role (Bat/Bowl/AR/WK), Nationality, Price

- [ ] **Get historical IPL player stats (2008-2025)**
  - Source: Kaggle IPL datasets or Cricsheet (ball-by-ball CSV/JSON)
  - Key datasets:
    - `sukhdayaldhanday/dream-11-fantasy-points-data-of-ipl-all-seasons` (Kaggle — pre-processed Dream11 points)
    - `patrickb1912/ipl-complete-dataset-20082020` (Kaggle — matches + deliveries)
    - `chaitu20/ipl-dataset2008-2025` (Kaggle — most recent)
    - Cricsheet.org for ball-by-ball data

- [ ] **Get Dream11 scoring system** — official point rules for T20
  - Runs: 1pt/run
  - Wicket: 25pts
  - Catch: 8pts, Stumping: 10pts, Direct runout: 10pts
  - Half-century bonus: 8-10pts, Century: 16pts
  - Maiden: 4pts
  - Strike rate & economy rate bonuses
  - Captain: 2x, Vice-captain: 1.5x
  - Source: dream11.com/fantasy-cricket/point-system

---

## Phase 2: Data Processing & Feature Engineering

- [ ] **Calculate historical fantasy points per player per match**
  - Apply Dream11 scoring rules to ball-by-ball data
  - Or use pre-calculated Kaggle dataset

- [ ] **Build player feature set:**
  - Batting position (opener, middle order, finisher)
  - Bowling type (pace, spin, medium)
  - Player role (pure bat, pure bowl, all-rounder, WK)
  - Venue/pitch type (batting-friendly, bowling-friendly, neutral)
  - Recent form (last 1-2 seasons weighted higher)
  - Career averages vs recent averages
  - Home/away splits
  - Phase-wise contribution (powerplay, middle, death)
  - Matchup data (vs pace/spin)

- [ ] **Pitch/venue classification**
  - Historical run rates per venue
  - Average scores per venue
  - Pace vs spin wicket percentages

---

## Phase 3: Expected Points Model (Python)

- [ ] **Choose modelling approach:**
  - Option A: Gradient Boosted Trees (XGBoost/LightGBM) — good for tabular data, handles mixed features
  - Option B: Random Forest — simpler, interpretable
  - Option C: Ensemble (blend of multiple models)
  - Option D: Bayesian approach — good for small sample sizes (new/young players)

- [ ] **Features for model:**
  - Player historical fantasy points (mean, median, std)
  - Role, batting position, bowling type
  - Venue characteristics
  - Opposition strength
  - Recent form (exponential decay weighting)
  - Season-level trends

- [ ] **Train/validate:**
  - Train on 2008-2024, validate on 2025
  - Cross-validation within training set
  - Evaluate: MAE, RMSE, rank correlation

- [ ] **Output: Expected points per player for IPL 2026**
  - Point estimate + confidence interval
  - Breakdown by role (batting pts, bowling pts, fielding pts)
  - Per-match expected + season aggregate expected

---

## Phase 4: Draft Cheat Sheet (Excel/CSV)

- [ ] **Final output spreadsheet with columns:**
  - Player name
  - Team
  - Role
  - Batting position
  - Bowling type
  - Nationality
  - Auction price
  - Historical avg fantasy pts (last 3 seasons)
  - Historical avg fantasy pts (career)
  - **Model predicted expected pts per match**
  - **Model predicted total season pts**
  - Confidence/uncertainty rating
  - Value score (expected pts / auction price)
  - Tier ranking (Tier 1 / 2 / 3)
  - Notes (injury, form, new team adjustment)

- [ ] **Sort/filter views:**
  - By expected points (overall)
  - By role (best batters, best bowlers, best all-rounders, best WKs)
  - By value (pts per crore)
  - By floor (minimum expected — safe picks)
  - By ceiling (maximum expected — boom/bust picks)

---

## Phase 5: Draft Strategy Notes

- [ ] **Positional scarcity analysis** — which roles have fewer elite options?
- [ ] **Stacking strategy** — players from same team (correlated upside)
- [ ] **Fixture analysis** — early season schedule, who plays weaker teams first?
- [ ] **Punt picks** — high-ceiling cheap players
- [ ] **Avoid list** — overpriced, injury-prone, out-of-form

---

## APIs & Data Sources Reference

| Source | URL | What it provides |
|--------|-----|-----------------|
| Kaggle (Dream11 pts) | kaggle.com/datasets/sukhdayaldhanday/... | Pre-processed fantasy points all seasons |
| Kaggle (IPL 2008-2025) | kaggle.com/datasets/chaitu20/... | Matches + deliveries |
| Cricsheet | cricsheet.org | Ball-by-ball data, CSV/JSON |
| CricAPI | cricapi.com | Free player stats API (100k hits/hr) |
| Roanuz Cricket API | cricketapi.com | Player stats + fantasy data |
| Dream11 scoring | dream11.com/fantasy-cricket/point-system | Official scoring rules |

---

## Tech Stack

- **Python** (Jupyter notebook or scripts)
- **pandas** for data wrangling
- **scikit-learn / XGBoost / LightGBM** for prediction model
- **openpyxl** for Excel output
- **matplotlib/seaborn** for quick viz if needed

---

*Draft is tomorrow. Prioritise Phase 1 (data) and Phase 4 (cheat sheet) over a perfect model.*
