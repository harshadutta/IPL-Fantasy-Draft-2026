"""Team preview with recalculated scoring."""
import json, pandas as pd, numpy as np

with open('data/processed/draft_teams.json') as f:
    teams = json.load(f)

actuals = pd.read_csv('data/processed/player_2025_actuals.csv', index_col='player')

SPECIAL = {
    'B Sai Sudharsan': 'B Sai Sudharsan', 'KL Rahul': 'KL Rahul',
    'Vaibhav Suryavanshi': 'V Suryavanshi', 'MS Dhoni': 'MS Dhoni',
    'Quinton de Kock': 'Q de Kock', 'Tim David': 'TH David',
    'Phil Salt': 'PD Salt', 'Suryakumar Yadav': 'SA Yadav',
    'Nicholas Pooran': 'N Pooran', 'Rishabh Pant': 'RR Pant',
    'Heinrich Klaasen': 'H Klaasen', 'Hardik Pandya': 'HH Pandya',
    'Riyan Parag': 'R Parag', 'Ajinkya Rahane': 'AM Rahane',
    'Prabhsimran Singh': 'P Simran Singh', 'Krunal Pandya': 'KH Pandya',
    'Deepak Chahar': 'DL Chahar', 'Prasidh Krishna': 'M Prasidh Krishna',
    'Vaibhav Arora': 'VG Arora', 'Josh Hazlewood': 'JR Hazlewood',
    'Trent Boult': 'TA Boult', 'Jasprit Bumrah': 'JJ Bumrah',
    'Ravindra Jadeja': 'RA Jadeja', 'Virat Kohli': 'V Kohli',
    'Rohit Sharma': 'RG Sharma', 'Jos Buttler': 'JC Buttler',
    'Shivam Dube': 'S Dube', 'Bhuvneshwar Kumar': 'B Kumar',
    'Marco Jansen': 'M Jansen', 'Liam Livingstone': 'LS Livingstone',
    'Travis Head': 'TM Head', 'Shreyas Iyer': 'SS Iyer',
    'Mitchell Marsh': 'MR Marsh', 'Axar Patel': 'AR Patel',
    'Dewald Brevis': 'D Brevis', 'Rajat Patidar': 'RM Patidar',
    'Sanju Samson': 'SV Samson', 'Angkrish Raghuvanshi': 'A Raghuvanshi',
    'Ayush Badoni': 'A Badoni', 'Mitchell Santner': 'MJ Santner',
    'Romario Shepherd': 'R Shepherd', 'Shubman Gill': 'Shubman Gill',
    'Yashasvi Jaiswal': 'YBK Jaiswal', 'Priyansh Arya': 'Priyansh Arya',
    'Shimron Hetmyer': 'SO Hetmyer', 'Tristan Stubbs': 'T Stubbs',
    'Nitish Rana': 'N Rana', 'Rahul Tewatia': 'R Tewatia',
    'Ramandeep Singh': 'RK Singh', 'Sunil Narine': 'SP Narine',
    'Aiden Markram': 'AK Markram', 'Kuldeep Yadav': 'Kuldeep Yadav',
    'Noor Ahmad': 'Noor Ahmad', 'Yuzvendra Chahal': 'YS Chahal',
    'Devdutt Padikkal': 'D Padikkal', 'Jofra Archer': 'JC Archer',
    'Harshal Patel': 'HV Patel', 'R Sai Kishore': 'R Sai Kishore',
    'Ruturaj Gaikwad': 'RD Gaikwad', 'David Miller': 'DA Miller',
    'Marcus Stoinis': 'MP Stoinis', 'Rinku Singh': 'Rinku Singh',
    'Ayush Mhatre': 'A Mhatre', 'Shashank Singh': 'Shashank Singh',
    'Mitchell Starc': 'Mitchell Starc', 'Ravi Bishnoi': 'Ravi Bishnoi',
    'Ryan Rickelton': 'RD Rickelton', 'Mohammed Shami': 'Mohammed Shami',
    'Washington Sundar': 'Washington Sundar', 'Abhishek Sharma': 'Abhishek Sharma',
    'Mohammed Siraj': 'Mohammed Siraj', 'Harsh Dubey': 'Harsh Dubey',
    'Azmatullah Omarzai': 'Azmatullah Omarzai', 'Sandeep Sharma': 'Sandeep Sharma',
    'Jacob Bethell': 'JG Bethell', 'Rachin Ravindra': 'R Ravindra',
    'Venkatesh Iyer': 'Venkatesh Iyer', 'Vipraj Nigam': 'V Nigam',
    'Swapnil Singh': 'Swapnil Singh',
}

def find_actual(name):
    if name in actuals.index:
        return actuals.loc[name]
    last = name.split()[-1].lower()
    first = name.split()[0][0].upper() if name.split() else ''
    for p in actuals.index:
        pl = p.split()[-1].lower()
        pf = p.split()[0][0].upper() if p.split() else ''
        if pl == last and pf == first:
            return actuals.loc[p]
    if name in SPECIAL and SPECIAL[name] in actuals.index:
        return actuals.loc[SPECIAL[name]]
    return None

# Process teams
results = []
for team in teams:
    name = team['name']
    is_user = team.get('is_user', False)
    total_2025 = 0
    sum_avg = 0
    cap_avg = 0
    vc_avg = 0
    players = []

    for p in team['players']:
        pname = p['player']
        tag = p.get('role_tag', '')
        a = find_actual(pname)

        avg = 0; total = 0; matches = 0; bowl_pct = 0; bat_pct = 0
        if a is not None:
            avg = round(a['avg_pts_2025'], 1)
            total = int(a['total_pts_2025'])
            matches = int(a['matches_2025'])
            if avg > 0:
                bowl_pct = round(a['avg_bowling_pts'] / avg * 100)
                bat_pct = round(a['avg_batting_pts'] / avg * 100)
            total_2025 += total
            sum_avg += avg

        if '2x' in tag: cap_avg = avg
        if '1.5x' in tag: vc_avg = avg

        players.append({
            'name': pname, 'team': p['ipl_team'], 'tag': tag,
            'avg': avg, 'total': total, 'matches': matches,
            'bowl_pct': bowl_pct, 'bat_pct': bat_pct,
        })

    effective = sum_avg + cap_avg + vc_avg * 0.5
    results.append({
        'team_name': name, 'is_user': is_user,
        'total_2025': total_2025, 'sum_avg': round(sum_avg, 1),
        'effective': round(effective, 1),
        'cap_name': next((p['name'] for p in players if '2x' in (p.get('tag') or '')), ''),
        'cap_avg': cap_avg,
        'vc_name': next((p['name'] for p in players if '1.5x' in (p.get('tag') or '')), ''),
        'vc_avg': vc_avg,
        'players': players,
    })

results.sort(key=lambda x: x['effective'], reverse=True)

print('=' * 80)
print('IPL 2026 FANTASY DRAFT PREVIEW - RECALCULATED SCORING')
print('Key changes: Dot balls +2 (was +1), Economy bonuses ~doubled,')
print('Wicket haul bonuses doubled, Duck penalty -4 (was -2)')
print('=' * 80)

print('\nTEAM POWER RANKINGS')
print('-' * 80)
print(f'{"Rk":>3} {"Team":>20} {"Sum Avg":>8} {"Cap (2x)":>18} {"VC (1.5x)":>18} {"Effective":>10} {"2025 Tot":>9}')
for i, t in enumerate(results):
    m = ' <<<' if t['is_user'] else ''
    print(f'{i+1:3d} {t["team_name"]:>20} {t["sum_avg"]:>8.1f} {t["cap_name"][:12]:>12}({t["cap_avg"]:.0f}) {t["vc_name"][:12]:>12}({t["vc_avg"]:.0f}) {t["effective"]:>10.1f} {t["total_2025"]:>9,}{m}')

# Detailed per team
for t in results:
    marker = ' ** YOUR TEAM **' if t['is_user'] else ''
    print(f'\n{"=" * 70}')
    print(f'#{results.index(t)+1} {t["team_name"]}{marker}')
    print(f'Sum avg: {t["sum_avg"]} | Effective/match: {t["effective"]} | 2025 total: {t["total_2025"]:,}')
    print(f'{"=" * 70}')
    print(f'{"Player":>25} {"IPL":>5} {"Tag":>8} {"Avg":>6} {"Total":>6} {"M":>3} {"Bat%":>5} {"Bwl%":>5}')
    print('-' * 70)

    for p in sorted(t['players'], key=lambda x: x['avg'], reverse=True):
        tag = p.get('tag', '') or ''
        avg_s = f'{p["avg"]:.0f}' if p['avg'] > 0 else '-'
        tot_s = f'{p["total"]}' if p['total'] > 0 else '-'
        m_s = f'{p["matches"]}' if p['matches'] > 0 else '-'
        bat_s = f'{p["bat_pct"]}%' if p['avg'] > 0 else '-'
        bwl_s = f'{p["bowl_pct"]}%' if p['avg'] > 0 else '-'
        print(f'{p["name"]:>25} {p["team"]:>5} {tag:>8} {avg_s:>6} {tot_s:>6} {m_s:>3} {bat_s:>5} {bwl_s:>5}')

if __name__ == '__main__':
    pass
