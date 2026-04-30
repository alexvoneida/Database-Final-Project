import pandas as pd

def GET_OPP_LAST_N_GAME_IDS(df: pd.DataFrame, n: int, game_id: str):
    '''Retrieve the last N games for a specific team before a given game ID.'''
    seen_game_ids = set()
    team_abb = df[df['GAME_ID'] == game_id]['MATCHUP'].values[0].split(' ')[-1]
    team_games = df[df['MATCHUP'].str.contains(team_abb)]
    team_games = team_games.sort_values(by='GAME_DATE', ascending=False)
    game_idx = team_games[team_games['GAME_ID'] == game_id].index
    team_games = team_games[team_games.index > game_idx[0]]
    for i in team_games.itertuples():
        if i.GAME_ID in seen_game_ids:
            team_games = team_games.drop(i.Index)
        else:
            seen_game_ids.add(i.GAME_ID)
    return team_games.head(n)['GAME_ID'].tolist()

def GET_OPP_LAST_N_GAME_STATS(game_stats_df: pd.DataFrame, game_ids_last5: list, matchup_abb: str) -> float:
    output = pd.DataFrame()
    for game_id in game_ids_last5:
        game_stats_by_id = game_stats_df[game_stats_df['GAME_ID'] == game_id]
        game_stats = game_stats_by_id[game_stats_by_id['TEAM_ABBREVIATION'] == matchup_abb]
        output = pd.concat([output, game_stats], ignore_index=True)
    return output

def OPP_OFF_RATING_last5(game_stats_df: pd.DataFrame) -> float:
    total = 0.0
    for row in game_stats_df.itertuples():
        total += row.offensiveRating
    return round(total / 5, 2)

def OPP_DEF_RATING_last5(game_stats_df: pd.DataFrame) -> float:
    total = 0.0
    for row in game_stats_df.itertuples():
        total += row.defensiveRating
    return round(total / 5, 2)

def OPP_REB_PCT_last5(game_stats_df: pd.DataFrame) -> float:
    total = 0.0
    for row in game_stats_df.itertuples():
        total += row.reboundPercentage
    return round(total / 5, 2)

def OPP_PACE_last5(game_stats_df: pd.DataFrame) -> float:
    total = 0.0
    for row in game_stats_df.itertuples():
        total += row.pace
    return round(total / 5, 2)

def main():
    features_df = pd.read_parquet('../parquet/player_game_logs_with_features_2025-26_FINAL.parquet')
    game_stats_df = pd.read_parquet('../parquet/game_stats_2025-26_FINAL.parquet')
    
    game_stats_df = game_stats_df.sort_values(['TEAM_ABBREVIATION', 'GAME_ID'])

    stats_cols = ['offensiveRating', 'defensiveRating', 'reboundPercentage', 'pace']
    window_size = 5

    game_stats_df[[f'{c}_last{window_size}' for c in stats_cols]] = (
        game_stats_df.groupby('TEAM_ABBREVIATION')[stats_cols]
        .apply(lambda x: x.shift().rolling(window_size, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    
    features_df = features_df.merge(
        game_stats_df[
            ['GAME_ID', 'TEAM_ABBREVIATION', 'offensiveRating_last5', 'defensiveRating_last5', 'pace_last5']
        ],
        on=['GAME_ID', 'TEAM_ABBREVIATION'],
        how='left'
    )
    
    opp_stats = game_stats_df.copy()
    opp_stats.rename(
        columns={
            'TEAM_ABBREVIATION':'OPP_ABBREVIATION',
            'offensiveRating_last5':'OPP_offensiveRating_last5',
            'defensiveRating_last5':'OPP_defensiveRating_last5',
            'pace_last5':'OPP_pace_last5'
        },
        inplace=True
    )
    
    features_df = features_df.merge(
        opp_stats[['GAME_ID', 'OPP_ABBREVIATION', 'OPP_offensiveRating_last5', 'OPP_defensiveRating_last5', 'OPP_pace_last5']],
        left_on=['GAME_ID', 'OPP_ABBREVIATION'],
        right_on=['GAME_ID', 'OPP_ABBREVIATION'],
        how='left'
    )
    
    features_df.to_parquet('../parquet/final_database_2025-26.parquet')

if __name__ == '__main__':
    main()