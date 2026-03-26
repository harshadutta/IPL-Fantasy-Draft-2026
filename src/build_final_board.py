"""
Build the final draft board merging 2025 performance data with 2026 squad/XI predictions.
"""

import pandas as pd
import numpy as np


def get_last_name(name):
    parts = name.strip().split()
    return parts[-1].lower() if parts else ''


def build_name_map(board_names, xi_names):
    """Match XI names (full) to board names (Cricsheet format)."""
    board_by_last = {}
    for n in board_names:
        ln = get_last_name(n)
        board_by_last.setdefault(ln, []).append(n)

    matches = {}
    for xi_name in xi_names:
        ln = get_last_name(xi_name)
        if ln in board_by_last:
            candidates = board_by_last[ln]
            if len(candidates) == 1:
                matches[xi_name] = candidates[0]
            else:
                xi_first = xi_name.split()[0][0].upper() if xi_name.split() else ''
                for c in candidates:
                    c_first = c.split()[0][0].upper() if c.split() else ''
                    if c_first == xi_first:
                        matches[xi_name] = c
                        break

    # Manual corrections for known mismatches
    manual = {
        'Ashutosh Sharma': 'Ashutosh Sharma',  # prevent wrong match
    }
    matches.update(manual)

    return matches


def estimate_new_player_pts(role, batting_pos, overs_bowled, role_avgs):
    """Estimate fantasy points for a player without 2025 IPL data."""
    # Use role-archetype averages from descriptive analysis
    if overs_bowled >= 3 and batting_pos <= 5 and batting_pos > 0:
        return role_avgs.get('Batting All-Rounder', 45)
    elif overs_bowled >= 3:
        return role_avgs.get('Pure Bowler', 40)
    elif overs_bowled >= 1 and batting_pos >= 6:
        return role_avgs.get('Bowling All-Rounder', 35)
    elif batting_pos <= 2 and batting_pos > 0:
        return role_avgs.get('Pure Opener', 50)
    elif batting_pos == 3:
        return role_avgs.get('Top-Order Bat', 45)
    elif batting_pos <= 5:
        return role_avgs.get('Middle-Order Bat', 40)
    elif batting_pos <= 7:
        return role_avgs.get('Lower-Order Bat', 35)
    else:
        return 30  # tail


def main():
    # Load data
    board = pd.read_csv('output/draft_board.csv', index_col='player')
    xis = pd.read_csv('data/processed/predicted_starting_xis.csv')
    squads = pd.read_csv('data/processed/ipl_2026_squads.csv')

    # Role archetype averages from descriptive analysis
    role_avgs = {
        'Batting All-Rounder': 77,
        'Pure Opener': 72,
        'Top-Order Bat': 66,
        'Pure Bowler': 54,
        'Middle-Order Bat': 52,
        'Bowling All-Rounder': 42,
        'Lower-Order Bat': 42,
    }

    # Build name mapping
    board_names = board.index.tolist()
    xi_names = xis['player'].tolist()
    name_map = build_name_map(board_names, xi_names)

    # Build final board: start with XI players
    rows = []
    for _, xi_row in xis.iterrows():
        xi_name = xi_row['player']
        cricsheet_name = name_map.get(xi_name)

        if cricsheet_name and cricsheet_name in board.index:
            # Returning player — use 2025 data
            b = board.loc[cricsheet_name]
            pred_pts = b['pred_pts_per_match']
            floor_p10 = b['floor_p10']
            ceiling_p90 = b['ceiling_p90']
            std_pts = b['std_pts']
            cv = b['cv']
            dud_rate = b['dud_rate']
            boom_rate = b['boom_rate']
            matches_2025 = b['matches']
            is_returning = True
            role_2025 = b['modal_role']
        else:
            # New player — estimate from context
            bp = xi_row['expected_batting_pos']
            ov = xi_row['expected_overs_bowled']
            pred_pts = estimate_new_player_pts(xi_row['role'], bp, ov, role_avgs)
            floor_p10 = pred_pts * 0.3  # rough estimate
            ceiling_p90 = pred_pts * 2.0
            std_pts = pred_pts * 0.5
            cv = 0.5
            dud_rate = 0.3
            boom_rate = 0.2
            matches_2025 = 0
            is_returning = False
            role_2025 = 'New'

        # Games confidence: use 2025 data if returning, else use XI prediction
        gc = xi_row['games_confidence']
        if is_returning and matches_2025 > 0:
            # Use actual 2025 games played ratio to estimate 2026 games
            # Cap at 14 league matches
            max_team_matches = 17  # max any team played in 2025
            games_ratio = matches_2025 / max_team_matches
            expected_games = min(14, max(5, round(games_ratio * 14)))
            # Override confidence based on actual data
            if games_ratio >= 0.8:
                gc = 'High'
            elif games_ratio >= 0.5:
                gc = 'Medium'
            else:
                gc = 'Low'
        else:
            if gc == 'High':
                expected_games = 14
            elif gc == 'Medium':
                expected_games = 11
            else:
                expected_games = 7

        # Discount new players slightly (uncertainty penalty)
        if not is_returning:
            pred_pts = pred_pts * 0.85  # 15% uncertainty discount

        pred_season = pred_pts * expected_games

        rows.append({
            'player': xi_name,
            'team_2026': xi_row['team'],
            'role_2026': xi_row['role'],
            'nationality': xi_row['nationality'],
            'expected_batting_pos': xi_row['expected_batting_pos'],
            'expected_overs_bowled': xi_row['expected_overs_bowled'],
            'pred_pts_per_match': round(pred_pts, 1),
            'pred_season_total': round(pred_season, 0),
            'floor_p10': round(floor_p10, 1),
            'ceiling_p90': round(ceiling_p90, 1),
            'consistency_std': round(std_pts, 1),
            'cv': round(cv, 2),
            'dud_rate': round(dud_rate, 2),
            'boom_rate': round(boom_rate, 2),
            'games_confidence': gc,
            'expected_games': expected_games,
            'matches_2025': int(matches_2025),
            'is_returning': is_returning,
            'notes': xi_row.get('notes', ''),
        })

    final = pd.DataFrame(rows)
    final = final.sort_values('pred_season_total', ascending=False)
    final['rank'] = range(1, len(final) + 1)

    # Cliff detection
    final['prev_total'] = final['pred_season_total'].shift(1)
    final['gap'] = final['prev_total'] - final['pred_season_total']
    final['pct_drop'] = (final['gap'] / final['prev_total'] * 100).round(1)
    final['cliff'] = final['gap'] > 40

    # Tier assignment
    def assign_tier(total):
        if total >= 1000: return 1
        elif total >= 800: return 2
        elif total >= 650: return 3
        elif total >= 530: return 4
        elif total >= 430: return 5
        elif total >= 350: return 6
        else: return 7

    final['tier'] = final['pred_season_total'].apply(assign_tier)

    # Reorder columns
    col_order = [
        'rank', 'tier', 'cliff', 'gap', 'player', 'team_2026', 'role_2026', 'nationality',
        'expected_batting_pos', 'expected_overs_bowled',
        'pred_pts_per_match', 'pred_season_total',
        'floor_p10', 'ceiling_p90', 'consistency_std',
        'dud_rate', 'boom_rate',
        'expected_games', 'games_confidence',
        'is_returning', 'matches_2025', 'notes',
    ]
    final = final[col_order]

    # Save
    final.to_csv('output/draft_board_final.csv', index=False)
    print(f"Final draft board: {len(final)} players")
    print()

    # Display
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 25)
    display_cols = ['rank', 'tier', 'cliff', 'player', 'team_2026', 'role_2026',
                    'expected_batting_pos', 'expected_overs_bowled',
                    'pred_pts_per_match', 'pred_season_total',
                    'floor_p10', 'ceiling_p90', 'games_confidence', 'is_returning']
    print(final[display_cols].to_string(index=False))

    print()
    print("=== TIER SUMMARY ===")
    tier_sum = final.groupby('tier').agg(
        n_players=('rank', 'count'),
        avg_pts=('pred_season_total', 'mean'),
        top=('pred_season_total', 'max'),
        bottom=('pred_season_total', 'min'),
    ).round(0)
    print(tier_sum.to_string())

    print()
    print("=== CLIFFS (gaps > 40 pts) ===")
    cliff_rows = final[final['cliff'] == True]
    for _, row in cliff_rows.iterrows():
        print(f"  Rank {int(row['rank'])}: {row['player']} ({row['pred_season_total']:.0f}) - gap of {row.get('gap', 0):.0f} pts from above")


if __name__ == '__main__':
    main()
