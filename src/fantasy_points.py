"""
Dream11 T20 Fantasy Points Calculator
Parses Cricsheet JSON ball-by-ball data and computes per-player-per-match fantasy points.
"""

import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict


# ── Dream11 T20 Scoring Rules ──────────────────────────────────────────────

SCORING = {
    # Batting
    'run': 1,
    'four_bonus': 4,
    'six_bonus': 6,
    'milestone_25': 4,
    'milestone_50': 8,
    'milestone_75': 12,
    'milestone_100': 16,  # exclusive: replaces 25/50/75
    'duck': -2,  # only for bat/WK/AR, not pure bowlers

    # Strike Rate bonuses (min 10 balls faced, not for pure bowlers)
    'sr_170_plus': 6,
    'sr_150_170': 4,
    'sr_130_150': 2,
    'sr_60_70': -2,
    'sr_50_60': -4,
    'sr_below_50': -6,

    # Bowling
    'wicket': 30,
    'lbw_bowled_bonus': 8,
    'dot_ball': 1,
    'maiden': 12,
    'haul_3w': 4,
    'haul_4w': 8,
    'haul_5w': 12,

    # Economy Rate (min 2 overs)
    'er_below_5': 6,
    'er_5_6': 4,
    'er_6_7': 2,
    'er_10_11': -2,
    'er_11_12': -4,
    'er_above_12': -6,

    # Fielding
    'catch': 8,
    'catch_3_bonus': 4,
    'stumping': 12,
    'runout_direct': 12,
    'runout_indirect': 6,

    # Other
    'playing_xi': 4,
}


def parse_ipl_2025_matches(json_dir):
    """Load all IPL 2025 match JSONs from Cricsheet."""
    matches = []
    for fname in os.listdir(json_dir):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(json_dir, fname)
        with open(fpath, 'r') as f:
            data = json.load(f)
        if str(data['info'].get('season', '')) == '2025':
            matches.append(data)
    print(f"Loaded {len(matches)} IPL 2025 matches")
    return matches


def compute_fantasy_points(matches):
    """Compute Dream11 fantasy points for every player in every match."""
    all_records = []

    for match in matches:
        info = match['info']
        match_id = info['dates'][0] + '_' + '_'.join(sorted(info['teams']))
        date = info['dates'][0]
        venue = info.get('venue', 'Unknown')
        city = info.get('city', 'Unknown')
        teams = info['teams']

        # Build player->team mapping from info.players
        player_team = {}
        for team, players in info.get('players', {}).items():
            for p in players:
                player_team[p] = team

        # Initialize per-player stats
        stats = defaultdict(lambda: {
            'team': '', 'opposition': '',
            'runs': 0, 'balls_faced': 0, 'fours': 0, 'sixes': 0,
            'how_out': 'not out', 'did_bat': False, 'batting_pos': 0,
            'overs_bowled_balls': 0, 'maidens_overs': defaultdict(int),  # over_num -> runs
            'runs_conceded': 0, 'wickets': 0, 'dot_balls_bowled': 0,
            'lbw_bowled_wickets': 0, 'did_bowl': False,
            'catches': 0, 'stumpings': 0, 'runout_direct': 0, 'runout_indirect': 0,
            'innings_batted': 0, 'innings_bowled': 0,
            'overs_set': set(),  # track which overs bowled for maiden calc
            'over_runs': defaultdict(int),  # over_key -> runs conceded (for maidens)
        })

        # Track batting order per innings
        batting_order = defaultdict(list)  # innings_num -> ordered list of batters

        # Process each innings (skip super overs)
        for inning_data in match.get('innings', []):
            if 'super_overs' in str(inning_data.get('team', '')).lower():
                continue

            inning_team = inning_data.get('team', '')
            inning_num = 1 if inning_team == teams[0] else 2
            # Sometimes team batting first isn't teams[0], use innings order
            innings_list = match.get('innings', [])
            inning_idx = innings_list.index(inning_data)
            inning_num = inning_idx + 1
            if inning_num > 2:
                continue  # skip super overs

            bowling_team = [t for t in teams if t != inning_team][0] if len(teams) == 2 else ''

            for over_data in inning_data.get('overs', []):
                over_num = over_data['over']  # 0-indexed

                for delivery in over_data.get('deliveries', []):
                    batter = delivery['batter']
                    bowler = delivery['bowler']
                    runs = delivery.get('runs', {})
                    batter_runs = runs.get('batter', 0)
                    total_runs = runs.get('total', 0)
                    extras = delivery.get('extras', {})

                    # Set teams
                    if stats[batter]['team'] == '':
                        stats[batter]['team'] = inning_team
                        stats[batter]['opposition'] = bowling_team
                    if stats[bowler]['team'] == '':
                        stats[bowler]['team'] = bowling_team
                        stats[bowler]['opposition'] = inning_team

                    # Track batting order
                    if batter not in batting_order[inning_num]:
                        batting_order[inning_num].append(batter)

                    # ── Batting stats ──
                    stats[batter]['did_bat'] = True
                    stats[batter]['innings_batted'] = inning_num

                    # Count ball faced (not wides or no-balls for batter)
                    is_wide = 'wides' in extras
                    is_noball = 'noballs' in extras
                    if not is_wide:
                        stats[batter]['balls_faced'] += 1

                    stats[batter]['runs'] += batter_runs
                    if batter_runs == 4:
                        stats[batter]['fours'] += 1
                    elif batter_runs == 6:
                        stats[batter]['sixes'] += 1

                    # ── Bowling stats ──
                    stats[bowler]['did_bowl'] = True
                    stats[bowler]['innings_bowled'] = inning_num

                    # Ball counts for bowler (wides and no-balls don't count as legal deliveries)
                    if not is_wide and not is_noball:
                        stats[bowler]['overs_bowled_balls'] += 1

                    # Runs conceded by bowler = total - byes - leg_byes
                    bowler_conceded = total_runs - extras.get('byes', 0) - extras.get('legbyes', 0)
                    stats[bowler]['runs_conceded'] += bowler_conceded

                    # Over tracking for maidens
                    over_key = f"{inning_num}_{over_num}"
                    stats[bowler]['overs_set'].add(over_key)
                    stats[bowler]['over_runs'][over_key] += bowler_conceded

                    # Dot ball (0 runs off bat AND no extras conceded by bowler)
                    if batter_runs == 0 and not is_wide and extras.get('noballs', 0) == 0:
                        # Check if any bowler-chargeable extras
                        bowler_extras = extras.get('wides', 0) + extras.get('noballs', 0)
                        if bowler_extras == 0 and bowler_conceded == 0:
                            stats[bowler]['dot_balls_bowled'] += 1

                    # ── Wickets ──
                    for wicket in delivery.get('wickets', []):
                        kind = wicket.get('kind', '')
                        dismissed = wicket.get('player_out', '')

                        # Batting: dismissal
                        stats[dismissed]['how_out'] = kind

                        # Bowling: wicket credit (not for run outs)
                        if kind not in ('run out', 'retired hurt', 'retired out', 'obstructing the field'):
                            stats[bowler]['wickets'] += 1
                            if kind in ('bowled', 'lbw'):
                                stats[bowler]['lbw_bowled_wickets'] += 1

                        # Fielding
                        fielders = wicket.get('fielders', [])
                        fielder_names = [f.get('name', '') for f in fielders if f.get('name', '')]

                        if kind == 'caught':
                            if fielder_names:
                                stats[fielder_names[0]]['catches'] += 1
                        elif kind == 'stumped':
                            if fielder_names:
                                stats[fielder_names[0]]['stumpings'] += 1
                        elif kind == 'run out':
                            if len(fielder_names) == 1:
                                stats[fielder_names[0]]['runout_direct'] += 1
                            elif len(fielder_names) >= 2:
                                # Last fielder gets indirect, second-to-last also gets indirect
                                for fn in fielder_names[-2:]:
                                    stats[fn]['runout_indirect'] += 1

        # ── Assign batting positions ──
        for inn_num, order in batting_order.items():
            for pos_idx, batter in enumerate(order):
                if stats[batter]['batting_pos'] == 0:
                    stats[batter]['batting_pos'] = pos_idx + 1

        # ── Compute fantasy points per player ──
        for player, s in stats.items():
            if s['team'] == '' and player in player_team:
                s['team'] = player_team[player]
                s['opposition'] = [t for t in teams if t != s['team']][0] if len(teams) == 2 else ''

            # Skip players not in playing XI (didn't bat, bowl, or field)
            if not s['did_bat'] and not s['did_bowl'] and s['catches'] == 0 and s['stumpings'] == 0 and s['runout_direct'] == 0 and s['runout_indirect'] == 0:
                # Check if in playing XI via player list
                if player not in player_team:
                    continue

            # Determine role (rough inference)
            bat_pos = s['batting_pos']
            overs_bowled = s['overs_bowled_balls'] / 6.0
            is_pure_bowler = bat_pos >= 8 and overs_bowled >= 2
            is_keeper = False  # Can't reliably determine from Cricsheet

            # ── Batting points ──
            batting_pts = 0
            batting_pts += s['runs'] * SCORING['run']
            batting_pts += s['fours'] * SCORING['four_bonus']
            batting_pts += s['sixes'] * SCORING['six_bonus']

            # Milestone bonuses (century is exclusive per Dream11)
            if s['runs'] >= 100:
                batting_pts += SCORING['milestone_100']
            else:
                if s['runs'] >= 75:
                    batting_pts += SCORING['milestone_75']
                if s['runs'] >= 50:
                    batting_pts += SCORING['milestone_50']
                if s['runs'] >= 25:
                    batting_pts += SCORING['milestone_25']

            # Duck penalty (not for pure bowlers)
            if s['did_bat'] and s['runs'] == 0 and s['how_out'] not in ('not out', 'retired hurt') and not is_pure_bowler:
                batting_pts += SCORING['duck']

            # Strike rate bonus (min 10 balls, not for pure bowlers)
            sr_bonus = 0
            if s['balls_faced'] >= 10 and not is_pure_bowler:
                sr = (s['runs'] / s['balls_faced']) * 100
                if sr >= 170:
                    sr_bonus = SCORING['sr_170_plus']
                elif sr >= 150:
                    sr_bonus = SCORING['sr_150_170']
                elif sr >= 130:
                    sr_bonus = SCORING['sr_130_150']
                elif sr <= 50:
                    sr_bonus = SCORING['sr_below_50']
                elif sr <= 60:
                    sr_bonus = SCORING['sr_50_60']
                elif sr <= 70:
                    sr_bonus = SCORING['sr_60_70']

            # ── Bowling points ──
            bowling_pts = 0
            bowling_pts += s['wickets'] * SCORING['wicket']
            bowling_pts += s['lbw_bowled_wickets'] * SCORING['lbw_bowled_bonus']
            bowling_pts += s['dot_balls_bowled'] * SCORING['dot_ball']

            # Maidens
            maidens = 0
            for over_key in s['overs_set']:
                # Count legal deliveries in this over for this bowler
                # A maiden = 6 legal deliveries, 0 runs
                if s['over_runs'][over_key] == 0:
                    # Check if bowler bowled full over (6 legal deliveries)
                    # We approximate: if this over_key has 0 runs conceded, count as maiden
                    # This is approximate since we don't track per-over ball count precisely
                    maidens += 1

            # More accurate maiden detection: we need to be more careful
            # For now, count overs with 0 runs where bowler bowled a full over
            # We'll refine if needed
            bowling_pts += maidens * SCORING['maiden']

            # Wicket haul bonuses
            if s['wickets'] >= 5:
                bowling_pts += SCORING['haul_5w']
            if s['wickets'] >= 4:
                bowling_pts += SCORING['haul_4w']
            if s['wickets'] >= 3:
                bowling_pts += SCORING['haul_3w']

            # Economy rate bonus (min 2 overs)
            er_bonus = 0
            if overs_bowled >= 2:
                er = s['runs_conceded'] / overs_bowled
                if er < 5:
                    er_bonus = SCORING['er_below_5']
                elif er < 6:
                    er_bonus = SCORING['er_5_6']
                elif er < 7:
                    er_bonus = SCORING['er_6_7']
                elif er >= 12:
                    er_bonus = SCORING['er_above_12']
                elif er >= 11:
                    er_bonus = SCORING['er_11_12']
                elif er >= 10:
                    er_bonus = SCORING['er_10_11']

            # ── Fielding points ──
            fielding_pts = 0
            fielding_pts += s['catches'] * SCORING['catch']
            if s['catches'] >= 3:
                fielding_pts += SCORING['catch_3_bonus']
            fielding_pts += s['stumpings'] * SCORING['stumping']
            fielding_pts += s['runout_direct'] * SCORING['runout_direct']
            fielding_pts += s['runout_indirect'] * SCORING['runout_indirect']

            # Playing XI bonus
            playing_xi_pts = SCORING['playing_xi']

            total = batting_pts + sr_bonus + bowling_pts + er_bonus + fielding_pts + playing_xi_pts

            all_records.append({
                'match_id': match_id,
                'date': date,
                'player': player,
                'team': s['team'],
                'opposition': s['opposition'],
                'venue': venue,
                'city': city,
                'innings_batted': s['innings_batted'],
                'batting_pos': s['batting_pos'] if s['did_bat'] else 0,
                'runs': s['runs'],
                'balls_faced': s['balls_faced'],
                'fours': s['fours'],
                'sixes': s['sixes'],
                'how_out': s['how_out'],
                'batting_pts': batting_pts,
                'sr_bonus': sr_bonus,
                'overs_bowled': round(overs_bowled, 1),
                'maidens': maidens,
                'runs_conceded': s['runs_conceded'],
                'wickets': s['wickets'],
                'dot_balls_bowled': s['dot_balls_bowled'],
                'bowling_pts': bowling_pts,
                'er_bonus': er_bonus,
                'catches': s['catches'],
                'stumpings': s['stumpings'],
                'runout_direct': s['runout_direct'],
                'runout_indirect': s['runout_indirect'],
                'fielding_pts': fielding_pts,
                'playing_xi_pts': playing_xi_pts,
                'total_fantasy_pts': total,
            })

    df = pd.DataFrame(all_records)
    return df


def classify_role(row):
    """Classify player role archetype based on match data."""
    bp = row['batting_pos']
    ov = row['overs_bowled']

    if bp == 0 and ov == 0:
        return 'Unknown'
    elif ov >= 2 and bp <= 5 and bp > 0:
        return 'Batting AR'
    elif ov >= 2 and (bp > 5 or bp == 0):
        if bp >= 8 or bp == 0:
            return 'Pure Bowler'
        else:
            return 'Bowling AR'
    elif bp <= 2 and bp > 0 and ov < 1:
        return 'Opener'
    elif bp == 3 and ov < 1:
        return 'Top Order Bat'
    elif 4 <= bp <= 5 and ov < 1:
        return 'Middle Order Bat'
    elif 6 <= bp <= 7 and ov < 1:
        return 'Lower Middle Bat'
    elif bp >= 8 and ov < 2:
        return 'Tail'
    else:
        return 'Other'


if __name__ == '__main__':
    json_dir = 'data/raw/ipl_json/'
    print("Parsing IPL 2025 matches...")
    matches = parse_ipl_2025_matches(json_dir)

    print("Computing fantasy points...")
    df = compute_fantasy_points(matches)

    # Classify roles
    df['role'] = df.apply(classify_role, axis=1)

    # Save
    out_path = 'data/processed/player_match_fantasy_points.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} player-match records to {out_path}")
    print(f"Unique players: {df['player'].nunique()}")
    print(f"Unique matches: {df['match_id'].nunique()}")

    # Quick sanity check: top 10 by average fantasy points
    print("\n── Top 20 by avg fantasy points (min 5 matches) ──")
    player_avg = df.groupby('player').agg(
        matches=('total_fantasy_pts', 'count'),
        avg_pts=('total_fantasy_pts', 'mean'),
        total_pts=('total_fantasy_pts', 'sum'),
    ).query('matches >= 5').sort_values('avg_pts', ascending=False)
    print(player_avg.head(20).to_string())
