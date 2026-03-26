"""
Descriptive Analysis & Fixed Effects Regression for IPL Fantasy Draft 2026
Produces summary statistics tables and regression outputs for the draft board.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import modelling libraries
try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def load_data():
    df = pd.read_csv('data/processed/player_match_fantasy_points.csv')
    print(f"Loaded {len(df)} player-match records, {df['player'].nunique()} players, {df['match_id'].nunique()} matches")
    return df


def batting_pos_bucket(pos):
    if pos == 0:
        return 'Did not bat'
    elif pos <= 2:
        return '1-2 (Opener)'
    elif pos == 3:
        return '3 (One-down)'
    elif pos <= 5:
        return '4-5 (Middle)'
    elif pos <= 7:
        return '6-7 (Lower-middle)'
    else:
        return '8-11 (Tail)'


def overs_bucket(overs):
    if overs == 0:
        return '0 (No bowling)'
    elif overs <= 2:
        return '0.1-2 overs'
    elif overs <= 3:
        return '2.1-3 overs'
    else:
        return '3.1-4 overs'


def role_archetype(row):
    bp = row['batting_pos']
    ov = row['overs_bowled']
    if bp == 0 and ov == 0:
        return 'Sub/Unused'
    elif ov >= 2 and 1 <= bp <= 5:
        return 'Batting All-Rounder'
    elif ov >= 3 and (bp > 5 or bp == 0):
        return 'Pure Bowler'
    elif ov >= 1 and 6 <= bp <= 8:
        return 'Bowling All-Rounder'
    elif 1 <= bp <= 2 and ov < 1:
        return 'Pure Opener'
    elif bp == 3 and ov < 1:
        return 'Top-Order Bat'
    elif 4 <= bp <= 5 and ov < 1:
        return 'Middle-Order Bat'
    elif 6 <= bp <= 7 and ov < 2:
        return 'Lower-Order Bat'
    elif bp >= 8 and ov < 3:
        return 'Tail/Part-Timer'
    else:
        return 'Other'


def analysis_by_role(df):
    """3a: Points by role archetype."""
    df['role_archetype'] = df.apply(role_archetype, axis=1)
    exclude = ['Sub/Unused', 'Other']
    role_df = df[~df['role_archetype'].isin(exclude)]

    stats = role_df.groupby('role_archetype').agg(
        count=('total_fantasy_pts', 'count'),
        mean_pts=('total_fantasy_pts', 'mean'),
        median_pts=('total_fantasy_pts', 'median'),
        std_pts=('total_fantasy_pts', 'std'),
        floor_p10=('total_fantasy_pts', lambda x: np.percentile(x, 10)),
        ceiling_p90=('total_fantasy_pts', lambda x: np.percentile(x, 90)),
        avg_batting=('batting_pts', 'mean'),
        avg_bowling=('bowling_pts', 'mean'),
        avg_fielding=('fielding_pts', 'mean'),
    ).sort_values('mean_pts', ascending=False)

    stats['batting_pct'] = (stats['avg_batting'] / stats['mean_pts'] * 100).round(1)
    stats['bowling_pct'] = (stats['avg_bowling'] / stats['mean_pts'] * 100).round(1)
    stats['fielding_pct'] = (stats['avg_fielding'] / stats['mean_pts'] * 100).round(1)

    print("\n=== POINTS BY ROLE ARCHETYPE ===")
    print(stats.round(1).to_string())
    return stats


def analysis_by_batting_pos(df):
    """3b: Points by batting position."""
    bat_df = df[df['batting_pos'] > 0].copy()
    bat_df['bat_pos_bucket'] = bat_df['batting_pos'].apply(batting_pos_bucket)

    stats = bat_df.groupby('bat_pos_bucket').agg(
        count=('total_fantasy_pts', 'count'),
        mean_pts=('total_fantasy_pts', 'mean'),
        median_pts=('total_fantasy_pts', 'median'),
        avg_balls_faced=('balls_faced', 'mean'),
        avg_runs=('runs', 'mean'),
        avg_fours=('fours', 'mean'),
        avg_sixes=('sixes', 'mean'),
    ).round(1)

    # Order
    order = ['1-2 (Opener)', '3 (One-down)', '4-5 (Middle)', '6-7 (Lower-middle)', '8-11 (Tail)']
    stats = stats.reindex(order)

    print("\n=== POINTS BY BATTING POSITION ===")
    print(stats.to_string())
    return stats


def analysis_by_bowling_workload(df):
    """3c: Points by bowling workload."""
    df_copy = df.copy()
    df_copy['overs_bucket'] = df_copy['overs_bowled'].apply(overs_bucket)

    stats = df_copy.groupby('overs_bucket').agg(
        count=('total_fantasy_pts', 'count'),
        mean_pts=('total_fantasy_pts', 'mean'),
        median_pts=('total_fantasy_pts', 'median'),
        avg_wickets=('wickets', 'mean'),
        avg_dots=('dot_balls_bowled', 'mean'),
        avg_bowling_pts=('bowling_pts', 'mean'),
    ).round(1)

    order = ['0 (No bowling)', '0.1-2 overs', '2.1-3 overs', '3.1-4 overs']
    stats = stats.reindex(order)

    print("\n=== POINTS BY BOWLING WORKLOAD ===")
    print(stats.to_string())
    return stats


def analysis_by_venue(df):
    """3d: Points by venue."""
    stats = df.groupby('venue').agg(
        matches=('match_id', 'nunique'),
        mean_pts=('total_fantasy_pts', 'mean'),
        avg_batting=('batting_pts', 'mean'),
        avg_bowling=('bowling_pts', 'mean'),
        avg_fielding=('fielding_pts', 'mean'),
    ).sort_values('mean_pts', ascending=False).round(1)

    stats['type'] = np.where(stats['avg_batting'] > stats['avg_bowling'] * 1.3, 'Batting-friendly',
                    np.where(stats['avg_bowling'] > stats['avg_batting'] * 0.9, 'Bowling-friendly', 'Balanced'))

    # Only venues with 2+ matches
    stats = stats[stats['matches'] >= 2]

    print("\n=== POINTS BY VENUE (min 2 matches) ===")
    print(stats.to_string())
    return stats


def analysis_by_innings(df):
    """3e: Points by innings."""
    bat_df = df[df['batting_pos'] > 0].copy()

    stats = bat_df.groupby('innings_batted').agg(
        count=('total_fantasy_pts', 'count'),
        mean_pts=('total_fantasy_pts', 'mean'),
        avg_batting=('batting_pts', 'mean'),
        avg_bowling=('bowling_pts', 'mean'),
    ).round(1)

    print("\n=== POINTS BY INNINGS ===")
    print(stats.to_string())
    return stats


def consistency_analysis(df):
    """3h: Player consistency analysis."""
    player_stats = df.groupby('player').agg(
        matches=('total_fantasy_pts', 'count'),
        mean_pts=('total_fantasy_pts', 'mean'),
        median_pts=('total_fantasy_pts', 'median'),
        std_pts=('total_fantasy_pts', 'std'),
        floor_p10=('total_fantasy_pts', lambda x: np.percentile(x, 10)),
        ceiling_p90=('total_fantasy_pts', lambda x: np.percentile(x, 90)),
        total_pts=('total_fantasy_pts', 'sum'),
        avg_batting=('batting_pts', 'mean'),
        avg_bowling=('bowling_pts', 'mean'),
        avg_fielding=('fielding_pts', 'mean'),
        modal_role=('role', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'),
    )

    player_stats['cv'] = (player_stats['std_pts'] / player_stats['mean_pts']).round(2)
    player_stats['dud_rate'] = df.groupby('player')['total_fantasy_pts'].apply(lambda x: (x < 15).mean()).round(2)
    player_stats['boom_rate'] = df.groupby('player')['total_fantasy_pts'].apply(lambda x: (x > 60).mean()).round(2)
    player_stats['batting_pct'] = (player_stats['avg_batting'] / player_stats['mean_pts'] * 100).round(1)
    player_stats['bowling_pct'] = (player_stats['avg_bowling'] / player_stats['mean_pts'] * 100).round(1)

    # Filter to min 5 matches
    player_stats = player_stats[player_stats['matches'] >= 5].sort_values('mean_pts', ascending=False)

    print("\n=== PLAYER CONSISTENCY (min 5 matches, top 30) ===")
    cols = ['matches', 'mean_pts', 'median_pts', 'std_pts', 'floor_p10', 'ceiling_p90', 'cv', 'dud_rate', 'boom_rate', 'modal_role']
    print(player_stats[cols].head(30).round(1).to_string())

    print("\n=== MOST CONSISTENT (lowest CV, min 5 matches, avg > 40) ===")
    consistent = player_stats[player_stats['mean_pts'] > 40].sort_values('cv').head(15)
    print(consistent[cols].round(1).to_string())

    print("\n=== HIGHEST FLOOR (P10, min 5 matches) ===")
    print(player_stats.sort_values('floor_p10', ascending=False)[cols].head(15).round(1).to_string())

    print("\n=== HIGHEST CEILING (P90, min 5 matches) ===")
    print(player_stats.sort_values('ceiling_p90', ascending=False)[cols].head(15).round(1).to_string())

    return player_stats


def run_fe_regression(df):
    """Phase 4a: Fixed effects regression."""
    # Filter to players with 5+ matches
    counts = df.groupby('player')['match_id'].count()
    valid_players = counts[counts >= 5].index
    reg_df = df[df['player'].isin(valid_players)].copy()

    reg_df['bat_pos_bucket'] = reg_df['batting_pos'].apply(batting_pos_bucket)
    reg_df['overs_bucket'] = reg_df['overs_bowled'].apply(overs_bucket)

    print(f"\n=== FIXED EFFECTS REGRESSION ===")
    print(f"Sample: {len(reg_df)} obs, {reg_df['player'].nunique()} players")

    if HAS_STATSMODELS:
        # Use statsmodels for proper FE regression
        # Create dummies manually to extract FE estimates
        formula = 'total_fantasy_pts ~ C(player) + C(venue) + C(bat_pos_bucket) + C(overs_bucket) + C(innings_batted)'

        try:
            model = ols(formula, data=reg_df).fit()
            print(f"R-squared: {model.rsquared:.3f}")
            print(f"Adj R-squared: {model.rsquared_adj:.3f}")
            print(f"Observations: {model.nobs:.0f}")

            # Extract venue effects
            venue_effects = {k.replace('C(venue)[T.', '').rstrip(']'): v
                           for k, v in model.params.items() if 'venue' in k}
            print("\nVenue Fixed Effects (top/bottom):")
            ve_sorted = sorted(venue_effects.items(), key=lambda x: x[1], reverse=True)
            for v, e in ve_sorted[:5]:
                print(f"  +{e:.1f}  {v}")
            print("  ...")
            for v, e in ve_sorted[-5:]:
                print(f"  {e:.1f}  {v}")

            # Extract batting position effects
            pos_effects = {k.replace('C(bat_pos_bucket)[T.', '').rstrip(']'): v
                          for k, v in model.params.items() if 'bat_pos' in k}
            print("\nBatting Position Effects:")
            for p, e in sorted(pos_effects.items(), key=lambda x: x[1], reverse=True):
                print(f"  {e:+.1f}  {p}")

            # Extract overs effects
            ov_effects = {k.replace('C(overs_bucket)[T.', '').rstrip(']'): v
                         for k, v in model.params.items() if 'overs_bucket' in k}
            print("\nBowling Overs Effects:")
            for o, e in sorted(ov_effects.items(), key=lambda x: x[1], reverse=True):
                print(f"  {e:+.1f}  {o}")

            # Innings effect
            inn_effects = {k.replace('C(innings_batted)[T.', '').rstrip(']'): v
                          for k, v in model.params.items() if 'innings' in k}
            print("\nInnings Effects:")
            for i, e in inn_effects.items():
                print(f"  {e:+.1f}  Innings {i}")

            # Extract player fixed effects
            player_fes = {}
            intercept = model.params.get('Intercept', 0)
            for k, v in model.params.items():
                if k.startswith('C(player)[T.'):
                    pname = k.replace('C(player)[T.', '').rstrip(']')
                    player_fes[pname] = v + intercept
            # The reference player gets just the intercept
            all_players_in_model = reg_df['player'].unique()
            for p in all_players_in_model:
                if p not in player_fes:
                    player_fes[p] = intercept

            fe_df = pd.DataFrame([
                {'player': p, 'player_fe': fe} for p, fe in player_fes.items()
            ]).sort_values('player_fe', ascending=False)

            print("\nTop 20 Player Fixed Effects:")
            for _, row in fe_df.head(20).iterrows():
                print(f"  {row['player_fe']:+.1f}  {row['player']}")

            return model, fe_df

        except Exception as e:
            print(f"Regression failed: {e}")
            print("Falling back to simple averages as pseudo-FEs")
            fe_df = reg_df.groupby('player')['total_fantasy_pts'].mean().reset_index()
            fe_df.columns = ['player', 'player_fe']
            fe_df = fe_df.sort_values('player_fe', ascending=False)
            return None, fe_df
    else:
        print("statsmodels not available, using simple averages as pseudo-FEs")
        fe_df = reg_df.groupby('player')['total_fantasy_pts'].mean().reset_index()
        fe_df.columns = ['player', 'player_fe']
        fe_df = fe_df.sort_values('player_fe', ascending=False)
        return None, fe_df


def run_tree_model(df):
    """Phase 4c: Light gradient boosted tree for interaction discovery."""
    if not HAS_SKLEARN:
        print("\nsklearn not available, skipping tree model")
        return None

    # Prepare features
    reg_df = df.copy()
    reg_df['bat_pos_bucket'] = reg_df['batting_pos'].apply(batting_pos_bucket)
    reg_df['overs_bucket'] = reg_df['overs_bowled'].apply(overs_bucket)

    # Encode categoricals
    le_player = LabelEncoder()
    le_venue = LabelEncoder()
    le_batpos = LabelEncoder()
    le_overs = LabelEncoder()

    reg_df['player_enc'] = le_player.fit_transform(reg_df['player'])
    reg_df['venue_enc'] = le_venue.fit_transform(reg_df['venue'])
    reg_df['batpos_enc'] = le_batpos.fit_transform(reg_df['bat_pos_bucket'])
    reg_df['overs_enc'] = le_overs.fit_transform(reg_df['overs_bucket'])

    features = ['player_enc', 'venue_enc', 'batpos_enc', 'overs_enc', 'innings_batted']
    X = reg_df[features].values
    y = reg_df['total_fantasy_pts'].values

    # Train with moderate regularization
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        min_samples_leaf=10, subsample=0.8, random_state=42
    )
    gbr.fit(X, y)

    # Feature importance
    importances = dict(zip(features, gbr.feature_importances_))
    print("\n=== TREE MODEL FEATURE IMPORTANCE ===")
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {imp:.3f}  {feat}")

    train_score = gbr.score(X, y)
    print(f"\nTrain R-squared: {train_score:.3f}")

    return gbr


def generate_draft_board(df, player_stats, fe_df=None):
    """Phase 6: Generate final draft board."""

    # Use player_stats as base
    board = player_stats.copy()

    # Merge player FEs if available
    if fe_df is not None and len(fe_df) > 0:
        board = board.merge(fe_df[['player', 'player_fe']], left_index=True, right_on='player', how='left')
        board = board.set_index('player')
        board['pred_pts_per_match'] = board['player_fe'].fillna(board['mean_pts'])
    else:
        board['pred_pts_per_match'] = board['mean_pts']

    # Estimate season total (assume 14 league matches, adjusted by games played ratio)
    # Use their 2025 match count to estimate reliability
    max_team_matches = df.groupby('team')['match_id'].nunique().max()
    board['games_ratio'] = board['matches'] / max_team_matches
    board['expected_games'] = np.clip(board['games_ratio'] * 14, 5, 14).round(0)
    board['pred_season_total'] = (board['pred_pts_per_match'] * board['expected_games']).round(0)

    # Games confidence
    board['games_confidence'] = np.where(board['games_ratio'] >= 0.8, 'High',
                                np.where(board['games_ratio'] >= 0.5, 'Medium', 'Low'))

    # Sort by predicted season total
    board = board.sort_values('pred_season_total', ascending=False)

    # Rank
    board['rank'] = range(1, len(board) + 1)

    # Cliff detection (>8% drop from previous player)
    board['prev_total'] = board['pred_season_total'].shift(1)
    board['pct_drop'] = ((board['prev_total'] - board['pred_season_total']) / board['prev_total'] * 100).round(1)
    board['cliff'] = board['pct_drop'] > 8

    # Tier assignment based on cliffs
    tier = 1
    tiers = []
    for _, row in board.iterrows():
        if row['cliff']:
            tier += 1
        tiers.append(tier)
    board['tier'] = tiers

    # Points breakdown percentages
    board['batting_pts_pct'] = board['batting_pct']
    board['bowling_pts_pct'] = board['bowling_pct']

    # Select output columns
    output_cols = [
        'rank', 'tier', 'cliff', 'pct_drop', 'matches', 'modal_role',
        'pred_pts_per_match', 'pred_season_total',
        'mean_pts', 'median_pts', 'std_pts',
        'floor_p10', 'ceiling_p90', 'cv',
        'dud_rate', 'boom_rate',
        'expected_games', 'games_confidence',
        'batting_pts_pct', 'bowling_pts_pct',
    ]

    result = board[[c for c in output_cols if c in board.columns]].copy()
    result.index.name = 'player'

    return result


def main():
    os.makedirs('output/descriptives', exist_ok=True)
    os.makedirs('output/model_diagnostics', exist_ok=True)

    df = load_data()

    # ── Phase 3: Descriptive Analysis ──
    print("\n" + "="*60)
    print("PHASE 3: DESCRIPTIVE ANALYSIS")
    print("="*60)

    role_stats = analysis_by_role(df)
    role_stats.to_csv('output/descriptives/points_by_role.csv')

    batpos_stats = analysis_by_batting_pos(df)
    batpos_stats.to_csv('output/descriptives/points_by_batting_position.csv')

    bowling_stats = analysis_by_bowling_workload(df)
    bowling_stats.to_csv('output/descriptives/points_by_bowling_workload.csv')

    venue_stats = analysis_by_venue(df)
    venue_stats.to_csv('output/descriptives/points_by_venue.csv')

    innings_stats = analysis_by_innings(df)
    innings_stats.to_csv('output/descriptives/points_by_innings.csv')

    player_stats = consistency_analysis(df)
    player_stats.to_csv('output/descriptives/player_consistency.csv')

    # ── Phase 4: Regression & Model ──
    print("\n" + "="*60)
    print("PHASE 4: FIXED EFFECTS REGRESSION & TREE MODEL")
    print("="*60)

    model, fe_df = run_fe_regression(df)
    if fe_df is not None:
        fe_df.to_csv('output/model_diagnostics/player_fixed_effects.csv', index=False)

    tree = run_tree_model(df)

    # ── Phase 6: Draft Board ──
    print("\n" + "="*60)
    print("PHASE 6: DRAFT BOARD")
    print("="*60)

    draft_board = generate_draft_board(df, player_stats, fe_df)

    # Save
    draft_board.to_csv('output/draft_board.csv')
    print(f"\nDraft board saved to output/draft_board.csv ({len(draft_board)} players)")

    print("\n=== TOP 40 DRAFT BOARD ===")
    display_cols = ['rank', 'tier', 'cliff', 'pred_pts_per_match', 'pred_season_total',
                    'floor_p10', 'ceiling_p90', 'games_confidence', 'modal_role']
    display_cols = [c for c in display_cols if c in draft_board.columns]
    print(draft_board[display_cols].head(40).to_string())

    # Tier summary
    print("\n=== TIER SUMMARY ===")
    tier_summary = draft_board.groupby('tier').agg(
        n_players=('rank', 'count'),
        avg_season_pts=('pred_season_total', 'mean'),
        min_season_pts=('pred_season_total', 'min'),
        max_season_pts=('pred_season_total', 'max'),
    ).round(0)
    print(tier_summary.to_string())

    print("\n=== CLIFF LOCATIONS ===")
    cliffs = draft_board[draft_board['cliff']]
    if len(cliffs) > 0:
        for idx, row in cliffs.iterrows():
            print(f"  After rank {row['rank']-1} -> rank {row['rank']} ({idx}): {row['pct_drop']:.1f}% drop")
    else:
        print("  No major cliffs detected (>8% threshold)")


if __name__ == '__main__':
    main()
