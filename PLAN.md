# IPL Fantasy Draft 2026 - Analytical Plan

## Objective

Build a data-driven **draft board** (ranked player list with tiers and cliff detection) to maximize total fantasy points across the IPL 2026 season. No roster composition constraints — pure points maximization.

## Scoring System

Using Dream11/My11Circle T20 scoring (both are identical):

| Category | Action | Points |
|----------|--------|--------|
| Batting | Run | +1 |
| | Four bonus | +4 |
| | Six bonus | +6 |
| | 25-run bonus | +4 |
| | 50-run bonus | +8 |
| | 75-run bonus | +12 |
| | Century bonus | +16 |
| | Duck (non-bowler) | -2 |
| SR Bonus | 170+ | +6 |
| (min 10 balls) | 150-170 | +4 |
| | 130-150 | +2 |
| | 60-70 | -2 |
| | 50-60 | -4 |
| | <50 | -6 |
| Bowling | Wicket (excl run out) | +30 |
| | LBW/Bowled bonus | +8 |
| | Dot ball | +1 |
| | Maiden | +12 |
| | 3W bonus | +4 |
| | 4W bonus | +8 |
| | 5W bonus | +12 |
| ER Bonus | <5 | +6 |
| (min 2 overs) | 5-6 | +4 |
| | 6-7 | +2 |
| | 10-11 | -2 |
| | 11-12 | -4 |
| | >12 | -6 |
| Fielding | Catch | +8 |
| | 3-catch bonus | +4 |
| | Stumping | +12 |
| | Run out (direct) | +12 |
| | Run out (not direct) | +6 |
| Other | Playing XI | +4 |
| | Captain | 2x |
| | Vice-Captain | 1.5x |

## Analytical Framework

Fantasy points = f(Player Fixed Effect, Context Features)

```
FP_im = alpha_i + X_im * beta + epsilon_im
```

- `alpha_i` = player intrinsic quality (skill, temperament)
- `X_im` = match context (venue, batting position, overs bowled, innings, opposition)
- For returning players: estimate alpha_i from 2025 IPL data
- For new players: use context-only estimates from the feature grid

## Phase 1: Data Acquisition (Issue #1)

### 1a. Cricsheet IPL 2025 Ball-by-Ball Data
- Download from cricsheet.org/downloads/ipl_json.zip
- Filter to 2025 season matches only
- Parse JSON into structured DataFrames (balls, matches, players)

### 1b. IPL 2026 Squad Rosters
- Collect all 10 team squads post-mega-auction
- Fields: Player, Team, Role, Nationality, Batting Style, Bowling Style
- Source: official IPL site, ESPNcricinfo, Wisden

### 1c. IPL 2026 Fixture List
- Match schedule with venues
- Map teams to home grounds

## Phase 2: Fantasy Points Calculator (Issue #2)

Build `src/fantasy_points.py` to compute Dream11 points from ball-by-ball data.

### Per-match, per-player aggregation:
- **Batting:** runs, balls faced, 4s, 6s, dismissal type -> batting pts + SR bonus + milestone bonus
- **Bowling:** overs, maidens, runs conceded, wickets, dot balls, dismissal types -> bowling pts + ER bonus + milestone bonus
- **Fielding:** catches, stumpings, run outs (direct/not) -> fielding pts

### Output: `data/processed/player_match_fantasy_points.csv`
Columns: match_id, player, team, role, batting_pos, batting_pts, bowling_pts, fielding_pts, total_fantasy_pts, balls_faced, overs_bowled, venue, innings, opposition

## Phase 3: Descriptive Analysis (Issue #3)

### 3a. Points by Role Archetype
- Pure batter (top 4, doesn't bowl)
- Pure bowler (bats 8-11, bowls 3-4 overs)
- Batting all-rounder (top 5, bowls 1-3 overs)
- Bowling all-rounder (bats 6-8, bowls 3-4 overs)
- WK-batter (keeps wicket, bats top 4)
- Distribution stats: mean, median, std, P10 (floor), P90 (ceiling)

### 3b. Points by Batting Position (1-11)
- Avg fantasy points, avg balls faced, boundary %
- Identify which positions produce the most points and why

### 3c. Points by Bowling Workload
- Group: 0 overs, 1-2 overs, 3-4 overs
- Avg fantasy points per group
- Dot ball rates, wicket rates by group

### 3d. Points by Venue
- Avg batting pts vs bowling pts per ground
- High-scoring vs low-scoring classification
- Spin-friendly vs pace-friendly (wicket type distribution)

### 3e. Points by Innings (1st vs 2nd)
- Batting pts: batting first vs chasing
- Bowling pts: bowling first vs defending

### 3f. Bowling Phase Analysis
- Powerplay (1-6), Middle (7-15), Death (16-20)
- Fantasy pts generated per phase by bowling type (pace/spin)

### 3g. Fielding Value
- Avg fielding pts: WK vs non-WK
- Which players accumulate most fielding pts?

### 3h. Consistency Analysis
- Per-player: mean, median, std, floor (P10), ceiling (P90)
- Dud rate (% matches < 15 pts), boom rate (% matches > 60 pts)

## Phase 4: Regression & Modelling (Issue #4)

### 4a. Match-Level Fixed Effects Regression
- Dependent variable: total_fantasy_pts per player per match
- Player fixed effects (alpha_i)
- Venue fixed effects
- Batting position fixed effects
- Bowling overs bucket fixed effects (0, 1-2, 3-4)
- Innings fixed effect (1st/2nd)
- Extract: player FE estimates + context coefficients

### 4b. Ball-Level Feature Extraction
- Player's dot ball % (bowling)
- Player's boundary % (batting)
- Phase contribution splits (PP/middle/death)
- Feed as enrichment features into match-level model

### 4c. Light XGBoost for Interaction Discovery
- Same match-level data, XGBoost with max_depth=3-4, heavy regularization
- Feature importance ranking
- SHAP values for interpretability
- Key question: which interactions matter most?

## Phase 5: IPL 2026 Squads & Starting XI Prediction (Issue #5)

### 5a. Roster Collection
- All 10 teams, full squad lists
- Player metadata: role, nationality, batting/bowling style

### 5b. Starting XI Prediction
- Each team gets 4 overseas slots, 7 Indian players
- Predict likely XI based on: squad composition, player roles, 2025 form
- Flag players with rotation risk (multiple overseas options competing)
- Output: confidence level for each player making the XI (High/Medium/Low)

### 5c. Fixture & Venue Mapping
- Map each team's home ground
- Count home vs away matches in league stage

## Phase 6: Draft Board Generation (Issue #6)

### 6a. Player Scoring Prediction
- Returning players: alpha_i (player FE) + 2026 context adjustments (new team? new venue?)
- New players: context-only estimate from feature grid + comparable player borrowing
- Multiply per-match estimate by expected games played

### 6b. Tier Assignment
- Cluster players into tiers using natural breaks in the distribution
- Use Jenks natural breaks or manual cliff detection (>5% drop to next player)

### 6c. Final Output: `output/draft_board.csv`
Columns:
- Rank, Tier, Player, Team, Role, Batting Position
- Predicted Pts/Match, Predicted Season Total
- Floor (P10), Ceiling (P90), Consistency (StdDev)
- Games Played Confidence (High/Med/Low)
- Cliff marker (TRUE if >5% drop from previous rank)
- Points breakdown (Batting/Bowling/Fielding %)

## Draft Strategy Notes

- **No roster constraints** -> pure BPA (Best Player Available) drafting
- **Snake/reverse snake** -> tier breaks matter more than exact ranking
- **Edge = information asymmetry** -> where your ranking differs from friends' vibes
- **Key insight from scoring:** dot balls (+1 each) make full-quota bowlers very valuable
- **Key insight from scoring:** all-rounders who genuinely bat AND bowl are two players in one pick
- **Diversification:** avoid >2 players from same IPL team to reduce correlation risk
