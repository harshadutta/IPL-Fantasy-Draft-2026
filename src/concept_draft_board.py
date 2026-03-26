"""
Final Draft Board: Map 2026 players into concept slots.
Combines researched playing XI probabilities with concept-level fantasy point estimates.
"""

import pandas as pd
import numpy as np
import os
import glob


# ── Concept coefficients from regression ──
# Batting context: baseline 39.6 + position + venue + innings
BAT_POS_EFFECT = {
    1: 30.7,   # Opener
    2: 30.7,   # Opener
    3: 26.1,   # One-down
    4: 8.5,    # Middle
    5: 8.5,    # Middle
    6: 0.0,    # Lower-middle (baseline)
    7: 0.0,    # Lower-middle
    8: -17.6,  # Tail
    9: -17.6,
    10: -17.6,
    11: -17.6,
}
BAT_BASELINE = 39.6

# Bowling concept: pts per over by type x phase (from ball-level analysis)
# A bowler bowling 4 overs gets roughly: PP overs + Middle overs + Death overs
# Weighted by typical phase distribution
BOWL_PTS_PER_OVER = {
    'Pace': 12.3,        # weighted across phases: ~2 death + 1 PP + 1 middle
    'Wrist Spin': 11.1,  # mostly middle + some death
    'Finger Spin': 10.5, # mostly middle
}
BOWL_BASELINE_PER_OVER = 11.0  # fallback

# Venue effects (batting)
VENUE_BAT_EFFECT = {
    'Batting-friendly': 4.4,
    'Bowling-friendly': -7.8,
    'Balanced': 0.0,
}

# Venue effects (bowling)
VENUE_BOWL_EFFECT = {
    'Batting-friendly': -6.3,
    'Bowling-friendly': 5.6,
    'Balanced': 0.0,
}

# Home grounds
TEAM_HOME = {
    'CSK': 'Bowling-friendly',   # Chepauk
    'MI': 'Balanced',            # Wankhede
    'RCB': 'Batting-friendly',   # Chinnaswamy
    'KKR': 'Batting-friendly',   # Eden Gardens
    'DC': 'Batting-friendly',    # Arun Jaitley
    'PBKS': 'Batting-friendly',  # Dharamsala/Mohali
    'RR': 'Batting-friendly',    # Sawai Mansingh
    'SRH': 'Balanced',           # Uppal
    'GT': 'Batting-friendly',    # Narendra Modi
    'LSG': 'Batting-friendly',   # Ekana
}

# Player fixed effects from 2025 (top players)
# Will be loaded from file


def load_researched_xis():
    """Load all researched playing XI CSVs."""
    pattern = 'data/processed/playing_xi_*.csv'
    files = glob.glob(pattern)
    if not files:
        print(f"WARNING: No playing XI files found matching {pattern}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"  Loaded {f}: {len(df)} players")
        except Exception as e:
            print(f"  Error loading {f}: {e}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"Total researched players: {len(combined)}")
        return combined
    return pd.DataFrame()


def load_player_fes():
    """Load player fixed effects from 2025 analysis."""
    fe_path = 'output/model_diagnostics/player_fixed_effects.csv'
    if os.path.exists(fe_path):
        fe = pd.read_csv(fe_path)
        return dict(zip(fe['player'], fe['player_fe']))
    return {}


def get_bowling_type(player_name, squads_df):
    """Get bowling type category for a player."""
    if squads_df is None or len(squads_df) == 0:
        return 'Unknown'
    match = squads_df[squads_df['player'] == player_name]
    if len(match) == 0:
        return 'Unknown'
    style = str(match.iloc[0].get('bowling_style', '')).lower()
    if 'fast' in style or 'medium' in style:
        return 'Pace'
    elif 'offbreak' in style or 'orthodox' in style:
        return 'Finger Spin'
    elif 'legbreak' in style or 'wrist' in style:
        return 'Wrist Spin'
    return 'Unknown'


def estimate_concept_pts(row, player_fes, name_map, squads_df):
    """Estimate fantasy points per match using concept framework."""
    player = row['player']
    team = row['team']
    bat_pos = row.get('expected_batting_pos', 6)
    overs = row.get('expected_overs_bowled', 0)
    prob_starting = row.get('prob_starting', 80) / 100.0

    # ── Batting concept pts ──
    pos_effect = BAT_POS_EFFECT.get(int(bat_pos), 0)

    # Venue: assume ~7 home, ~7 away. Average the venue effect.
    home_venue = TEAM_HOME.get(team, 'Balanced')
    home_bat_effect = VENUE_BAT_EFFECT.get(home_venue, 0)
    avg_bat_venue = home_bat_effect * 0.5  # half games at home, half away (avg = neutral)

    concept_bat_pts = BAT_BASELINE + pos_effect + avg_bat_venue

    # ── Bowling concept pts ──
    concept_bowl_pts = 0
    if overs > 0:
        bowl_type = get_bowling_type(player, squads_df)
        pts_per_over = BOWL_PTS_PER_OVER.get(bowl_type, BOWL_BASELINE_PER_OVER)
        home_bowl_effect = VENUE_BOWL_EFFECT.get(home_venue, 0)
        avg_bowl_venue = home_bowl_effect * 0.5
        concept_bowl_pts = (pts_per_over * overs) + avg_bowl_venue

    # ── Fielding + Playing XI ──
    fielding_est = 4  # avg ~4 pts from fielding
    playing_xi = 4

    concept_total = concept_bat_pts + concept_bowl_pts + fielding_est + playing_xi

    # ── Player fixed effect adjustment ──
    # If we have 2025 data, blend concept estimate with player FE
    cricsheet_name = name_map.get(player, player)
    player_fe = player_fes.get(cricsheet_name, None)

    if player_fe is not None:
        # Blend: 40% concept, 60% player FE (player skill matters more)
        blended_pts = 0.4 * concept_total + 0.6 * player_fe
        is_returning = True
    else:
        blended_pts = concept_total
        is_returning = False

    return {
        'concept_bat_pts': round(concept_bat_pts, 1),
        'concept_bowl_pts': round(concept_bowl_pts, 1),
        'concept_total': round(concept_total, 1),
        'player_fe_2025': round(player_fe, 1) if player_fe else None,
        'blended_pts_per_match': round(blended_pts, 1),
        'is_returning': is_returning,
    }


def build_name_map(squads_df, fe_names):
    """Map full player names to Cricsheet format names."""
    name_map = {}

    def get_last(n):
        return n.strip().split()[-1].lower() if n.strip().split() else ''

    fe_by_last = {}
    for n in fe_names:
        fe_by_last.setdefault(get_last(n), []).append(n)

    if squads_df is not None:
        for full_name in squads_df['player'].unique():
            ln = get_last(full_name)
            if ln in fe_by_last:
                candidates = fe_by_last[ln]
                if len(candidates) == 1:
                    name_map[full_name] = candidates[0]
                else:
                    fi = full_name.split()[0][0].upper() if full_name.split() else ''
                    for c in candidates:
                        ci = c.split()[0][0].upper() if c.split() else ''
                        if ci == fi:
                            name_map[full_name] = c
                            break

    return name_map


def main():
    print("Loading data...")
    xis = load_researched_xis()
    if len(xis) == 0:
        print("No researched XI data found. Exiting.")
        return

    player_fes = load_player_fes()
    print(f"Player FEs loaded: {len(player_fes)}")

    # Load 2025 actuals
    actuals_2025 = None
    if os.path.exists('data/processed/player_2025_actuals.csv'):
        actuals_2025 = pd.read_csv('data/processed/player_2025_actuals.csv', index_col='player')
        print(f"2025 actuals loaded: {len(actuals_2025)} players")

    squads_df = None
    if os.path.exists('data/processed/ipl_2026_squads.csv'):
        squads_df = pd.read_csv('data/processed/ipl_2026_squads.csv')

    # Build name mapping
    name_map = build_name_map(xis, list(player_fes.keys()))
    print(f"Name mappings: {len(name_map)}")

    # Estimate concept pts for each player
    results = []
    for _, row in xis.iterrows():
        est = estimate_concept_pts(row, player_fes, name_map, squads_df)
        result = row.to_dict()
        result.update(est)

        # Expected games
        prob_14 = row.get('prob_all_14', 50) / 100.0
        prob_start = row.get('prob_starting', 80) / 100.0
        # Expected games = P(starting) * 14 * adjustment for rotation
        expected_games = round(prob_start * 14 * (0.5 + 0.5 * prob_14), 1)
        expected_games = min(14, max(3, expected_games))
        result['expected_games'] = expected_games
        result['pred_season_total'] = round(result['blended_pts_per_match'] * expected_games, 0)

        # Merge 2025 actuals
        cricsheet_name = name_map.get(row['player'], row['player'])
        if actuals_2025 is not None and cricsheet_name in actuals_2025.index:
            a = actuals_2025.loc[cricsheet_name]
            result['matches_2025'] = int(a['matches_2025'])
            result['total_pts_2025'] = int(a['total_pts_2025'])
            result['avg_pts_2025'] = round(a['avg_pts_2025'], 1)
            result['median_pts_2025'] = round(a['median_pts_2025'], 1)
            result['floor_p10_2025'] = round(a['floor_p10_2025'], 1)
            result['ceiling_p90_2025'] = round(a['ceiling_p90_2025'], 1)
            result['avg_batting_pts_2025'] = round(a['avg_batting_pts'], 1)
            result['avg_bowling_pts_2025'] = round(a['avg_bowling_pts'], 1)
        else:
            result['matches_2025'] = 0
            result['total_pts_2025'] = None
            result['avg_pts_2025'] = None
            result['median_pts_2025'] = None
            result['floor_p10_2025'] = None
            result['ceiling_p90_2025'] = None
            result['avg_batting_pts_2025'] = None
            result['avg_bowling_pts_2025'] = None

        results.append(result)

    final = pd.DataFrame(results)
    final = final.sort_values('pred_season_total', ascending=False)
    final['rank'] = range(1, len(final) + 1)

    # Cliff detection
    final['prev_total'] = final['pred_season_total'].shift(1)
    final['gap'] = final['prev_total'] - final['pred_season_total']
    final['cliff'] = final['gap'] > 40

    # Tier assignment
    def assign_tier(total):
        if total >= 900: return 1
        elif total >= 750: return 2
        elif total >= 600: return 3
        elif total >= 500: return 4
        elif total >= 400: return 5
        elif total >= 300: return 6
        else: return 7

    final['tier'] = final['pred_season_total'].apply(assign_tier)

    # Select output columns
    out_cols = [
        'rank', 'tier', 'cliff', 'player', 'team', 'role', 'nationality',
        'expected_batting_pos', 'expected_overs_bowled',
        # 2025 actuals
        'matches_2025', 'total_pts_2025', 'avg_pts_2025', 'median_pts_2025',
        'floor_p10_2025', 'ceiling_p90_2025',
        'avg_batting_pts_2025', 'avg_bowling_pts_2025',
        # Concept model
        'concept_bat_pts', 'concept_bowl_pts', 'concept_total',
        'player_fe_2025', 'blended_pts_per_match', 'pred_season_total',
        'expected_games', 'prob_starting', 'prob_all_14',
        'is_returning', 'notes',
    ]
    out_cols = [c for c in out_cols if c in final.columns]
    output = final[out_cols]

    # Save
    output.to_csv('output/concept_draft_board.csv', index=False)
    print(f"\nSaved concept draft board: {len(output)} players")
    print(f"  output/concept_draft_board.csv")

    # Display
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 25)
    display = ['rank', 'tier', 'player', 'team', 'role',
               'expected_batting_pos', 'expected_overs_bowled',
               'avg_pts_2025', 'total_pts_2025', 'matches_2025',
               'concept_total', 'blended_pts_per_match',
               'pred_season_total', 'expected_games', 'prob_starting']
    display = [c for c in display if c in output.columns]
    print("\n=== CONCEPT DRAFT BOARD (ALL) ===")
    print(output[display].to_string(index=False))

    # Tier summary
    print("\n=== TIER SUMMARY ===")
    tier_sum = output.groupby('tier').agg(
        n=('rank', 'count'),
        avg_season=('pred_season_total', 'mean'),
        top=('pred_season_total', 'max'),
        bottom=('pred_season_total', 'min'),
    ).round(0)
    print(tier_sum.to_string())


if __name__ == '__main__':
    main()
