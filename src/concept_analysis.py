"""
Concept-Level Analysis: Where do fantasy points come from?
Decomposes fantasy points into contextual features WITHOUT player identity.
Uses ball-by-ball data for maximum granularity.
"""

import json
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict

try:
    from sklearn.tree import DecisionTreeRegressor, export_text
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    HAS_SM = True
except ImportError:
    HAS_SM = False


# ── Dream11 scoring constants ──
PTS_RUN = 1
PTS_FOUR = 4
PTS_SIX = 6
PTS_DOT_BOWLED = 1
PTS_WICKET = 30
PTS_LBW_BOWLED = 8
PTS_CATCH = 8
PTS_STUMPING = 12
PTS_RUNOUT_DIRECT = 12
PTS_RUNOUT_INDIRECT = 6


def classify_venue(venue):
    """Classify venue into type based on known characteristics."""
    batting_grounds = [
        'Narendra Modi Stadium', 'Eden Gardens', 'Arun Jaitley',
        'Sawai Mansingh', 'Dharamsala', 'Ekana', 'M Chinnaswamy',
    ]
    bowling_grounds = [
        'Chepauk', 'Chidambaram', 'Wankhede', 'Brabourne',
    ]
    for bg in batting_grounds:
        if bg.lower() in venue.lower():
            return 'Batting-friendly'
    for bg in bowling_grounds:
        if bg.lower() in venue.lower():
            return 'Bowling-friendly'
    return 'Balanced'


def phase_of_over(over_num):
    """0-indexed over to phase."""
    if over_num <= 5:
        return 'Powerplay (1-6)'
    elif over_num <= 14:
        return 'Middle (7-15)'
    else:
        return 'Death (16-20)'


def batting_pos_label(pos):
    if pos <= 2:
        return 'Opener (1-2)'
    elif pos == 3:
        return 'One-down (3)'
    elif pos <= 5:
        return 'Middle (4-5)'
    elif pos <= 7:
        return 'Lower-middle (6-7)'
    else:
        return 'Tail (8+)'


def build_bowling_type_map(squads_path, json_dir):
    """Build player -> bowling type mapping from squad data + inference."""
    bowling_map = {}

    # From squad CSV
    if os.path.exists(squads_path):
        sq = pd.read_csv(squads_path)
        # Map to broader categories
        for _, row in sq.iterrows():
            style = str(row.get('bowling_style', '')).lower()
            name = row['player']
            if 'fast' in style or 'medium' in style:
                bowling_map[name] = 'Pace'
            elif 'offbreak' in style or 'orthodox' in style:
                bowling_map[name] = 'Finger Spin'
            elif 'legbreak' in style or 'wrist' in style:
                bowling_map[name] = 'Wrist Spin'
            else:
                bowling_map[name] = 'Unknown'

    # For Cricsheet names, we need to map from the squad names
    # Build a last-name lookup for approximate matching
    last_name_map = {}
    for name, btype in list(bowling_map.items()):
        parts = name.strip().split()
        if parts:
            last = parts[-1].lower()
            last_name_map.setdefault(last, []).append((name, btype))

    return bowling_map, last_name_map


def resolve_bowling_type(bowler_name, bowling_map, last_name_map):
    """Try to find bowling type for a Cricsheet-format name."""
    if bowler_name in bowling_map:
        return bowling_map[bowler_name]

    # Try last name match
    parts = bowler_name.strip().split()
    if parts:
        last = parts[-1].lower()
        if last in last_name_map:
            candidates = last_name_map[last]
            if len(candidates) == 1:
                return candidates[0][1]
            # Try first initial match
            first_init = parts[0][0].upper() if parts[0] else ''
            for full_name, btype in candidates:
                fn_parts = full_name.split()
                if fn_parts and fn_parts[0][0].upper() == first_init:
                    return btype

    return 'Unknown'


def parse_ball_level_data(json_dir, bowling_map, last_name_map):
    """Parse all IPL 2025 matches into ball-level records."""
    all_batting_balls = []
    all_bowling_balls = []

    for fname in os.listdir(json_dir):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(json_dir, fname)
        with open(fpath) as f:
            data = json.load(f)
        if str(data['info'].get('season', '')) != '2025':
            continue

        info = data['info']
        match_id = info['dates'][0] + '_' + '_'.join(sorted(info['teams']))
        venue = info.get('venue', 'Unknown')
        venue_type = classify_venue(venue)
        teams = info['teams']

        for inning_idx, inning_data in enumerate(data.get('innings', [])):
            inning_num = inning_idx + 1
            if inning_num > 2:
                continue  # skip super overs

            batting_team = inning_data.get('team', '')
            bowling_team = [t for t in teams if t != batting_team][0] if len(teams) == 2 else ''

            # Track batting order
            batting_order = []

            for over_data in inning_data.get('overs', []):
                over_num = over_data['over']  # 0-indexed
                phase = phase_of_over(over_num)

                for delivery in over_data.get('deliveries', []):
                    batter = delivery['batter']
                    bowler = delivery['bowler']
                    runs = delivery.get('runs', {})
                    batter_runs = runs.get('batter', 0)
                    total_runs = runs.get('total', 0)
                    extras = delivery.get('extras', {})

                    is_wide = 'wides' in extras
                    is_noball = 'noballs' in extras

                    # Track batting position
                    if batter not in batting_order:
                        batting_order.append(batter)
                    bat_pos = batting_order.index(batter) + 1

                    # Bowling type
                    bowl_type = resolve_bowling_type(bowler, bowling_map, last_name_map)

                    # ── Batting ball record ──
                    if not is_wide:  # wides don't count as balls faced
                        is_four = batter_runs == 4
                        is_six = batter_runs == 6
                        is_dot_faced = (batter_runs == 0)

                        bat_pts_this_ball = batter_runs * PTS_RUN
                        if is_four:
                            bat_pts_this_ball += PTS_FOUR
                        if is_six:
                            bat_pts_this_ball += PTS_SIX

                        all_batting_balls.append({
                            'match_id': match_id,
                            'venue': venue,
                            'venue_type': venue_type,
                            'innings': inning_num,
                            'over': over_num + 1,  # 1-indexed for readability
                            'phase': phase,
                            'batter': batter,
                            'batting_team': batting_team,
                            'batting_pos': bat_pos,
                            'batting_pos_label': batting_pos_label(bat_pos),
                            'bowler_type_faced': bowl_type,
                            'runs': batter_runs,
                            'is_four': is_four,
                            'is_six': is_six,
                            'is_dot_faced': is_dot_faced,
                            'batting_pts': bat_pts_this_ball,
                        })

                    # ── Bowling ball record ──
                    if not is_wide and not is_noball:  # legal delivery
                        bowler_conceded = total_runs - extras.get('byes', 0) - extras.get('legbyes', 0)
                        is_dot = (bowler_conceded == 0)

                        # Check for wicket on this ball
                        wicket_on_ball = False
                        is_lbw_bowled = False
                        for wk in delivery.get('wickets', []):
                            kind = wk.get('kind', '')
                            if kind not in ('run out', 'retired hurt', 'retired out', 'obstructing the field'):
                                wicket_on_ball = True
                                if kind in ('bowled', 'lbw'):
                                    is_lbw_bowled = True

                        bowl_pts = 0
                        if is_dot:
                            bowl_pts += PTS_DOT_BOWLED
                        if wicket_on_ball:
                            bowl_pts += PTS_WICKET
                            if is_lbw_bowled:
                                bowl_pts += PTS_LBW_BOWLED

                        all_bowling_balls.append({
                            'match_id': match_id,
                            'venue': venue,
                            'venue_type': venue_type,
                            'innings': inning_num,
                            'over': over_num + 1,
                            'phase': phase,
                            'bowler': bowler,
                            'bowling_team': bowling_team,
                            'bowling_type': bowl_type,
                            'runs_conceded': bowler_conceded,
                            'is_dot': is_dot,
                            'is_wicket': wicket_on_ball,
                            'is_lbw_bowled': is_lbw_bowled,
                            'bowling_pts': bowl_pts,
                        })

    bat_df = pd.DataFrame(all_batting_balls)
    bowl_df = pd.DataFrame(all_bowling_balls)
    print(f"Batting balls: {len(bat_df)}")
    print(f"Bowling balls: {len(bowl_df)}")
    return bat_df, bowl_df


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def batting_concept_means(bat_df):
    """Conditional means for batting concepts."""
    print("\n" + "=" * 70)
    print("BATTING CONCEPTS: WHERE DO BATTING POINTS COME FROM?")
    print("=" * 70)

    # 1. By batting position
    print("\n--- Batting Points per BALL by Position ---")
    pos = bat_df.groupby('batting_pos_label').agg(
        balls=('batting_pts', 'count'),
        pts_per_ball=('batting_pts', 'mean'),
        runs_per_ball=('runs', 'mean'),
        four_rate=('is_four', 'mean'),
        six_rate=('is_six', 'mean'),
        dot_rate=('is_dot_faced', 'mean'),
    ).round(3)
    pos = pos.reindex(['Opener (1-2)', 'One-down (3)', 'Middle (4-5)',
                       'Lower-middle (6-7)', 'Tail (8+)'])
    print(pos.to_string())

    # 2. By venue type
    print("\n--- Batting Points per Ball by Venue Type ---")
    ven = bat_df.groupby('venue_type').agg(
        balls=('batting_pts', 'count'),
        pts_per_ball=('batting_pts', 'mean'),
        runs_per_ball=('runs', 'mean'),
        four_rate=('is_four', 'mean'),
        six_rate=('is_six', 'mean'),
    ).round(3)
    print(ven.to_string())

    # 3. By innings
    print("\n--- Batting Points per Ball by Innings ---")
    inn = bat_df.groupby('innings').agg(
        balls=('batting_pts', 'count'),
        pts_per_ball=('batting_pts', 'mean'),
        runs_per_ball=('runs', 'mean'),
    ).round(3)
    print(inn.to_string())

    # 4. By phase
    print("\n--- Batting Points per Ball by Phase ---")
    ph = bat_df.groupby('phase').agg(
        balls=('batting_pts', 'count'),
        pts_per_ball=('batting_pts', 'mean'),
        runs_per_ball=('runs', 'mean'),
        four_rate=('is_four', 'mean'),
        six_rate=('is_six', 'mean'),
    ).round(3)
    ph = ph.reindex(['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)'])
    print(ph.to_string())

    # 5. By bowling type faced
    print("\n--- Batting Points per Ball vs Bowling Type ---")
    bt = bat_df[bat_df['bowler_type_faced'] != 'Unknown'].groupby('bowler_type_faced').agg(
        balls=('batting_pts', 'count'),
        pts_per_ball=('batting_pts', 'mean'),
        runs_per_ball=('runs', 'mean'),
        four_rate=('is_four', 'mean'),
        six_rate=('is_six', 'mean'),
        dot_rate=('is_dot_faced', 'mean'),
    ).round(3)
    print(bt.to_string())

    # 6. KEY INTERACTIONS: Position x Phase
    print("\n--- Batting Pts/Ball: Position x Phase ---")
    pp = bat_df.groupby(['batting_pos_label', 'phase']).agg(
        balls=('batting_pts', 'count'),
        pts_per_ball=('batting_pts', 'mean'),
    ).round(3)
    pp_pivot = pp.reset_index().pivot(index='batting_pos_label', columns='phase', values='pts_per_ball')
    pp_pivot = pp_pivot.reindex(['Opener (1-2)', 'One-down (3)', 'Middle (4-5)',
                                  'Lower-middle (6-7)', 'Tail (8+)'])
    pp_pivot = pp_pivot[['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)']]
    print(pp_pivot.to_string())

    # 7. Position x Venue Type
    print("\n--- Batting Pts/Ball: Position x Venue Type ---")
    pv = bat_df.groupby(['batting_pos_label', 'venue_type']).agg(
        balls=('batting_pts', 'count'),
        pts_per_ball=('batting_pts', 'mean'),
    ).round(3)
    pv_pivot = pv.reset_index().pivot(index='batting_pos_label', columns='venue_type', values='pts_per_ball')
    pv_pivot = pv_pivot.reindex(['Opener (1-2)', 'One-down (3)', 'Middle (4-5)',
                                  'Lower-middle (6-7)', 'Tail (8+)'])
    print(pv_pivot.to_string())

    # 8. Position x Innings
    print("\n--- Batting Pts/Ball: Position x Innings ---")
    pi = bat_df.groupby(['batting_pos_label', 'innings']).agg(
        balls=('batting_pts', 'count'),
        pts_per_ball=('batting_pts', 'mean'),
    ).round(3)
    pi_pivot = pi.reset_index().pivot(index='batting_pos_label', columns='innings', values='pts_per_ball')
    pi_pivot = pi_pivot.reindex(['Opener (1-2)', 'One-down (3)', 'Middle (4-5)',
                                  'Lower-middle (6-7)', 'Tail (8+)'])
    pi_pivot.columns = ['Bat 1st', 'Chase']
    print(pi_pivot.to_string())

    return pos


def bowling_concept_means(bowl_df):
    """Conditional means for bowling concepts."""
    print("\n" + "=" * 70)
    print("BOWLING CONCEPTS: WHERE DO BOWLING POINTS COME FROM?")
    print("=" * 70)

    known = bowl_df[bowl_df['bowling_type'] != 'Unknown']

    # 1. By bowling type
    print("\n--- Bowling Points per Ball by Type ---")
    bt = known.groupby('bowling_type').agg(
        balls=('bowling_pts', 'count'),
        pts_per_ball=('bowling_pts', 'mean'),
        dot_rate=('is_dot', 'mean'),
        wicket_rate=('is_wicket', 'mean'),
        avg_conceded=('runs_conceded', 'mean'),
    ).round(4)
    print(bt.to_string())

    # 2. By phase
    print("\n--- Bowling Points per Ball by Phase ---")
    ph = bowl_df.groupby('phase').agg(
        balls=('bowling_pts', 'count'),
        pts_per_ball=('bowling_pts', 'mean'),
        dot_rate=('is_dot', 'mean'),
        wicket_rate=('is_wicket', 'mean'),
        avg_conceded=('runs_conceded', 'mean'),
    ).round(4)
    ph = ph.reindex(['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)'])
    print(ph.to_string())

    # 3. By venue type
    print("\n--- Bowling Points per Ball by Venue Type ---")
    vt = bowl_df.groupby('venue_type').agg(
        balls=('bowling_pts', 'count'),
        pts_per_ball=('bowling_pts', 'mean'),
        dot_rate=('is_dot', 'mean'),
        wicket_rate=('is_wicket', 'mean'),
    ).round(4)
    print(vt.to_string())

    # 4. By innings
    print("\n--- Bowling Points per Ball by Innings ---")
    inn = bowl_df.groupby('innings').agg(
        balls=('bowling_pts', 'count'),
        pts_per_ball=('bowling_pts', 'mean'),
        dot_rate=('is_dot', 'mean'),
        wicket_rate=('is_wicket', 'mean'),
    ).round(4)
    print(inn.to_string())

    # 5. KEY: Bowling Type x Phase
    print("\n--- Bowling Pts/Ball: Type x Phase ---")
    tp = known.groupby(['bowling_type', 'phase']).agg(
        balls=('bowling_pts', 'count'),
        pts_per_ball=('bowling_pts', 'mean'),
        dot_rate=('is_dot', 'mean'),
        wicket_rate=('is_wicket', 'mean'),
    ).round(4)
    tp_pivot = tp.reset_index().pivot(index='bowling_type', columns='phase', values='pts_per_ball')
    tp_pivot = tp_pivot[['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)']]
    print(tp_pivot.to_string())

    print("\n--- Dot Ball Rate: Type x Phase ---")
    dot_pivot = tp.reset_index().pivot(index='bowling_type', columns='phase', values='dot_rate')
    dot_pivot = dot_pivot[['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)']]
    print(dot_pivot.to_string())

    print("\n--- Wicket Rate: Type x Phase ---")
    wk_pivot = tp.reset_index().pivot(index='bowling_type', columns='phase', values='wicket_rate')
    wk_pivot = wk_pivot[['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)']]
    print(wk_pivot.to_string())

    # 6. Bowling Type x Venue Type
    print("\n--- Bowling Pts/Ball: Type x Venue ---")
    tv = known.groupby(['bowling_type', 'venue_type']).agg(
        balls=('bowling_pts', 'count'),
        pts_per_ball=('bowling_pts', 'mean'),
    ).round(4)
    tv_pivot = tv.reset_index().pivot(index='bowling_type', columns='venue_type', values='pts_per_ball')
    print(tv_pivot.to_string())

    # 7. Bowling Type x Innings
    print("\n--- Bowling Pts/Ball: Type x Innings ---")
    ti = known.groupby(['bowling_type', 'innings']).agg(
        balls=('bowling_pts', 'count'),
        pts_per_ball=('bowling_pts', 'mean'),
    ).round(4)
    ti_pivot = ti.reset_index().pivot(index='bowling_type', columns='innings', values='pts_per_ball')
    ti_pivot.columns = ['Bowl 1st', 'Bowl 2nd (defend)']
    print(ti_pivot.to_string())

    return bt


def per_match_aggregation(bat_df, bowl_df):
    """Aggregate ball-level to per-match-per-concept for regression."""
    print("\n" + "=" * 70)
    print("PER-MATCH CONCEPT AGGREGATION")
    print("=" * 70)

    # Batting: aggregate per batter per match
    bat_match = bat_df.groupby(['match_id', 'batter', 'batting_pos_label', 'venue_type', 'innings', 'batting_team']).agg(
        balls_faced=('batting_pts', 'count'),
        batting_pts=('batting_pts', 'sum'),
        runs=('runs', 'sum'),
        fours=('is_four', 'sum'),
        sixes=('is_six', 'sum'),
        dots_faced=('is_dot_faced', 'sum'),
    ).reset_index()

    # Add SR bonus estimate (simplified)
    bat_match['strike_rate'] = np.where(bat_match['balls_faced'] > 0,
                                         bat_match['runs'] / bat_match['balls_faced'] * 100, 0)
    bat_match['sr_bonus'] = 0
    bat_match.loc[(bat_match['balls_faced'] >= 10) & (bat_match['strike_rate'] >= 170), 'sr_bonus'] = 6
    bat_match.loc[(bat_match['balls_faced'] >= 10) & (bat_match['strike_rate'] >= 150) & (bat_match['strike_rate'] < 170), 'sr_bonus'] = 4
    bat_match.loc[(bat_match['balls_faced'] >= 10) & (bat_match['strike_rate'] >= 130) & (bat_match['strike_rate'] < 150), 'sr_bonus'] = 2
    bat_match.loc[(bat_match['balls_faced'] >= 10) & (bat_match['strike_rate'] <= 70) & (bat_match['strike_rate'] > 60), 'sr_bonus'] = -2
    bat_match.loc[(bat_match['balls_faced'] >= 10) & (bat_match['strike_rate'] <= 60) & (bat_match['strike_rate'] > 50), 'sr_bonus'] = -4
    bat_match.loc[(bat_match['balls_faced'] >= 10) & (bat_match['strike_rate'] <= 50), 'sr_bonus'] = -6

    # Milestone bonuses
    bat_match['milestone'] = 0
    bat_match.loc[bat_match['runs'] >= 100, 'milestone'] = 16
    bat_match.loc[(bat_match['runs'] >= 75) & (bat_match['runs'] < 100), 'milestone'] = 12 + 8 + 4
    bat_match.loc[(bat_match['runs'] >= 50) & (bat_match['runs'] < 75), 'milestone'] = 8 + 4
    bat_match.loc[(bat_match['runs'] >= 25) & (bat_match['runs'] < 50), 'milestone'] = 4

    bat_match['total_bat_pts'] = bat_match['batting_pts'] + bat_match['sr_bonus'] + bat_match['milestone']

    # Bowling: aggregate per bowler per match
    bowl_match = bowl_df.groupby(['match_id', 'bowler', 'bowling_type', 'venue_type', 'innings', 'bowling_team']).agg(
        balls_bowled=('bowling_pts', 'count'),
        bowling_pts=('bowling_pts', 'sum'),
        dots=('is_dot', 'sum'),
        wickets=('is_wicket', 'sum'),
        runs_conceded=('runs_conceded', 'sum'),
    ).reset_index()

    bowl_match['overs'] = bowl_match['balls_bowled'] / 6.0

    # Economy rate bonus
    bowl_match['er'] = np.where(bowl_match['overs'] > 0,
                                 bowl_match['runs_conceded'] / bowl_match['overs'], 99)
    bowl_match['er_bonus'] = 0
    bowl_match.loc[(bowl_match['overs'] >= 2) & (bowl_match['er'] < 5), 'er_bonus'] = 6
    bowl_match.loc[(bowl_match['overs'] >= 2) & (bowl_match['er'] >= 5) & (bowl_match['er'] < 6), 'er_bonus'] = 4
    bowl_match.loc[(bowl_match['overs'] >= 2) & (bowl_match['er'] >= 6) & (bowl_match['er'] < 7), 'er_bonus'] = 2
    bowl_match.loc[(bowl_match['overs'] >= 2) & (bowl_match['er'] >= 10) & (bowl_match['er'] < 11), 'er_bonus'] = -2
    bowl_match.loc[(bowl_match['overs'] >= 2) & (bowl_match['er'] >= 11) & (bowl_match['er'] < 12), 'er_bonus'] = -4
    bowl_match.loc[(bowl_match['overs'] >= 2) & (bowl_match['er'] >= 12), 'er_bonus'] = -6

    # Wicket haul bonuses
    bowl_match['haul_bonus'] = 0
    bowl_match.loc[bowl_match['wickets'] >= 3, 'haul_bonus'] += 4
    bowl_match.loc[bowl_match['wickets'] >= 4, 'haul_bonus'] += 8
    bowl_match.loc[bowl_match['wickets'] >= 5, 'haul_bonus'] += 12

    bowl_match['total_bowl_pts'] = bowl_match['bowling_pts'] + bowl_match['er_bonus'] + bowl_match['haul_bonus']

    print(f"Batting match records: {len(bat_match)}")
    print(f"Bowling match records: {len(bowl_match)}")

    return bat_match, bowl_match


def run_context_regression(bat_match, bowl_match):
    """FE regression without player identity — context only."""
    print("\n" + "=" * 70)
    print("CONTEXT-ONLY REGRESSION (no player identity)")
    print("=" * 70)

    if not HAS_SM:
        print("statsmodels not available, skipping regression")
        return

    # ── Batting regression ──
    print("\n--- BATTING: total_bat_pts ~ position + venue + innings ---")
    bm = bat_match[bat_match['balls_faced'] >= 5].copy()  # min 5 balls
    try:
        bat_model = ols('total_bat_pts ~ C(batting_pos_label) + C(venue_type) + C(innings)', data=bm).fit()
        print(f"R-squared: {bat_model.rsquared:.3f} (context explains {bat_model.rsquared*100:.1f}% of batting variance)")
        print(f"Observations: {int(bat_model.nobs)}")
        print("\nCoefficients:")
        for name, coef in bat_model.params.items():
            if name == 'Intercept':
                print(f"  Intercept (baseline): {coef:+.1f}")
            else:
                clean = name.replace('C(batting_pos_label)[T.', '').replace('C(venue_type)[T.', '').replace('C(innings)[T.', 'Innings ').rstrip(']')
                print(f"  {clean}: {coef:+.1f}")
    except Exception as e:
        print(f"Batting regression failed: {e}")

    # ── Bowling regression ──
    print("\n--- BOWLING: total_bowl_pts ~ type + phase_dominant + venue + innings ---")
    bw = bowl_match[(bowl_match['overs'] >= 2) & (bowl_match['bowling_type'] != 'Unknown')].copy()
    try:
        bowl_model = ols('total_bowl_pts ~ C(bowling_type) + C(venue_type) + C(innings)', data=bw).fit()
        print(f"R-squared: {bowl_model.rsquared:.3f} (context explains {bowl_model.rsquared*100:.1f}% of bowling variance)")
        print(f"Observations: {int(bowl_model.nobs)}")
        print("\nCoefficients:")
        for name, coef in bowl_model.params.items():
            if name == 'Intercept':
                print(f"  Intercept (baseline): {coef:+.1f}")
            else:
                clean = name.replace('C(bowling_type)[T.', '').replace('C(venue_type)[T.', '').replace('C(innings)[T.', 'Innings ').rstrip(']')
                print(f"  {clean}: {coef:+.1f}")
    except Exception as e:
        print(f"Bowling regression failed: {e}")


def run_concept_tree(bat_match, bowl_match):
    """Shallow tree model to discover context interactions."""
    if not HAS_SKLEARN:
        print("\nsklearn not available, skipping tree model")
        return

    print("\n" + "=" * 70)
    print("CONCEPT TREES (discovering interactions without player identity)")
    print("=" * 70)

    # ── Batting tree ──
    print("\n--- BATTING CONCEPT TREE ---")
    bm = bat_match[bat_match['balls_faced'] >= 5].copy()
    le_pos = LabelEncoder()
    le_ven = LabelEncoder()
    bm['pos_enc'] = le_pos.fit_transform(bm['batting_pos_label'])
    bm['ven_enc'] = le_ven.fit_transform(bm['venue_type'])

    X_bat = bm[['pos_enc', 'ven_enc', 'innings']].values
    y_bat = bm['total_bat_pts'].values

    tree_bat = DecisionTreeRegressor(max_depth=4, min_samples_leaf=30)
    tree_bat.fit(X_bat, y_bat)
    print(f"R-squared: {tree_bat.score(X_bat, y_bat):.3f}")

    feature_names = ['batting_position', 'venue_type', 'innings']
    tree_text = export_text(tree_bat, feature_names=feature_names, max_depth=4)
    print(tree_text)

    # ── Bowling tree ──
    print("\n--- BOWLING CONCEPT TREE ---")
    bw = bowl_match[(bowl_match['overs'] >= 2) & (bowl_match['bowling_type'] != 'Unknown')].copy()
    le_bt = LabelEncoder()
    le_vb = LabelEncoder()
    bw['bt_enc'] = le_bt.fit_transform(bw['bowling_type'])
    bw['vb_enc'] = le_vb.fit_transform(bw['venue_type'])

    X_bowl = bw[['bt_enc', 'vb_enc', 'innings']].values
    y_bowl = bw['total_bowl_pts'].values

    tree_bowl = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20)
    tree_bowl.fit(X_bowl, y_bowl)
    print(f"R-squared: {tree_bowl.score(X_bowl, y_bowl):.3f}")

    feature_names_b = ['bowling_type', 'venue_type', 'innings']
    tree_text_b = export_text(tree_bowl, feature_names=feature_names_b, max_depth=4)
    print(tree_text_b)


def build_concept_ranking(bat_match, bowl_match):
    """Build ranked concept tables — the main output."""
    print("\n" + "=" * 70)
    print("CONCEPT RANKINGS")
    print("=" * 70)

    # ── Batting concepts: position x venue_type x innings ──
    print("\n--- BATTING CONCEPT RANKING (position x venue x innings) ---")
    print("    Shows expected batting fantasy pts per match for each context slot")
    print("    (min 20 observations)")
    bat_concepts = bat_match[bat_match['balls_faced'] >= 3].groupby(
        ['batting_pos_label', 'venue_type', 'innings']
    ).agg(
        n=('total_bat_pts', 'count'),
        avg_bat_pts=('total_bat_pts', 'mean'),
        median_bat_pts=('total_bat_pts', 'median'),
        avg_balls_faced=('balls_faced', 'mean'),
        avg_sr=('strike_rate', 'mean'),
    ).round(1)

    bat_concepts = bat_concepts[bat_concepts['n'] >= 20].sort_values('avg_bat_pts', ascending=False)
    bat_concepts.reset_index(inplace=True)
    bat_concepts['innings_label'] = bat_concepts['innings'].map({1: '1st (bat first)', 2: '2nd (chase)'})
    bat_concepts['concept'] = bat_concepts['batting_pos_label'] + ' | ' + bat_concepts['venue_type'] + ' | ' + bat_concepts['innings_label']
    bat_concepts['bat_rank'] = range(1, len(bat_concepts) + 1)

    print(bat_concepts[['bat_rank', 'concept', 'n', 'avg_bat_pts', 'median_bat_pts', 'avg_balls_faced']].to_string(index=False))

    # ── Bowling concepts: type x venue_type x innings ──
    print("\n--- BOWLING CONCEPT RANKING (type x venue x innings) ---")
    print("    Shows expected bowling fantasy pts per match for each context slot")
    print("    (min 15 observations, min 2 overs)")
    bowl_concepts = bowl_match[
        (bowl_match['overs'] >= 2) & (bowl_match['bowling_type'] != 'Unknown')
    ].groupby(
        ['bowling_type', 'venue_type', 'innings']
    ).agg(
        n=('total_bowl_pts', 'count'),
        avg_bowl_pts=('total_bowl_pts', 'mean'),
        median_bowl_pts=('total_bowl_pts', 'median'),
        avg_overs=('overs', 'mean'),
        avg_wickets=('wickets', 'mean'),
        avg_dots=('dots', 'mean'),
        avg_er=('er', 'mean'),
    ).round(1)

    bowl_concepts = bowl_concepts[bowl_concepts['n'] >= 15].sort_values('avg_bowl_pts', ascending=False)
    bowl_concepts.reset_index(inplace=True)
    bowl_concepts['innings_label'] = bowl_concepts['innings'].map({1: '1st (bowl first)', 2: '2nd (defend)'})
    bowl_concepts['concept'] = bowl_concepts['bowling_type'] + ' | ' + bowl_concepts['venue_type'] + ' | ' + bowl_concepts['innings_label']
    bowl_concepts['bowl_rank'] = range(1, len(bowl_concepts) + 1)

    print(bowl_concepts[['bowl_rank', 'concept', 'n', 'avg_bowl_pts', 'median_bowl_pts', 'avg_overs', 'avg_wickets', 'avg_dots']].to_string(index=False))

    # Save
    bat_concepts.to_csv('output/batting_concept_ranking.csv', index=False)
    bowl_concepts.to_csv('output/bowling_concept_ranking.csv', index=False)

    return bat_concepts, bowl_concepts


def bowling_phase_deep_dive(bowl_df):
    """Deep dive into bowling by type x phase at ball level."""
    print("\n" + "=" * 70)
    print("BOWLING DEEP DIVE: Type x Phase (ball level)")
    print("=" * 70)

    known = bowl_df[bowl_df['bowling_type'] != 'Unknown']

    # Per-over estimates: if a bowler bowls one full over (6 balls) in this phase
    # what's the expected fantasy points from that over?
    print("\n--- Expected Fantasy Points per OVER by Type x Phase ---")
    tp = known.groupby(['bowling_type', 'phase']).agg(
        balls=('bowling_pts', 'count'),
        pts_per_ball=('bowling_pts', 'mean'),
        dot_rate=('is_dot', 'mean'),
        wicket_rate=('is_wicket', 'mean'),
        avg_conceded=('runs_conceded', 'mean'),
    )
    tp['pts_per_over'] = (tp['pts_per_ball'] * 6).round(1)
    tp['dots_per_over'] = (tp['dot_rate'] * 6).round(1)
    tp['wickets_per_over'] = (tp['wicket_rate'] * 6).round(2)
    tp['runs_per_over'] = (tp['avg_conceded'] * 6).round(1)

    result = tp[['balls', 'pts_per_over', 'dots_per_over', 'wickets_per_over', 'runs_per_over']]
    result = result.reset_index()

    # Pivot for readability
    for metric in ['pts_per_over', 'dots_per_over', 'wickets_per_over']:
        print(f"\n  {metric}:")
        pivot = result.pivot(index='bowling_type', columns='phase', values=metric)
        pivot = pivot[['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)']]
        print(pivot.to_string())

    # Rank all type x phase combinations
    print("\n--- RANKED: Bowling Type x Phase by Pts/Over ---")
    ranked = result[['bowling_type', 'phase', 'balls', 'pts_per_over', 'dots_per_over', 'wickets_per_over']].copy()
    ranked = ranked[ranked['balls'] >= 100].sort_values('pts_per_over', ascending=False)
    ranked['rank'] = range(1, len(ranked) + 1)
    print(ranked.to_string(index=False))


def main():
    os.makedirs('output', exist_ok=True)

    # Build bowling type mapping
    print("Building bowling type map...")
    bowling_map, last_name_map = build_bowling_type_map(
        'data/processed/ipl_2026_squads.csv',
        'data/raw/ipl_json/'
    )
    print(f"  Known bowling types: {len(bowling_map)} players")

    # Parse ball-level data
    print("\nParsing ball-by-ball data...")
    bat_df, bowl_df = parse_ball_level_data('data/raw/ipl_json/', bowling_map, last_name_map)

    # Save ball-level data
    bat_df.to_csv('data/processed/batting_balls_2025.csv', index=False)
    bowl_df.to_csv('data/processed/bowling_balls_2025.csv', index=False)

    # Check bowling type coverage
    known_pct = (bowl_df['bowling_type'] != 'Unknown').mean() * 100
    print(f"\nBowling type coverage: {known_pct:.1f}% of deliveries have known type")
    print(bowl_df['bowling_type'].value_counts().to_string())

    # ── Run all analyses ──

    # 1. Conditional means (cross-tabs)
    batting_concept_means(bat_df)
    bowling_concept_means(bowl_df)

    # 2. Deep dive into bowling by type x phase
    bowling_phase_deep_dive(bowl_df)

    # 3. Per-match aggregation
    bat_match, bowl_match = per_match_aggregation(bat_df, bowl_df)

    # 4. Context-only regression
    run_context_regression(bat_match, bowl_match)

    # 5. Concept trees
    run_concept_tree(bat_match, bowl_match)

    # 6. Concept rankings
    build_concept_ranking(bat_match, bowl_match)

    print("\n" + "=" * 70)
    print("DONE. Concept rankings saved to:")
    print("  output/batting_concept_ranking.csv")
    print("  output/bowling_concept_ranking.csv")
    print("=" * 70)


if __name__ == '__main__':
    main()
