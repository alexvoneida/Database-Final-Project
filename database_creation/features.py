import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from nba_api.stats.endpoints import boxscoreadvancedv3 as boxscore
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
import time

def MIN_last5(df_last5: pd.DataFrame) -> float:
    '''Calculate the average minutes played over the last 5 games for a specific player.'''
    if df_last5.empty:
        return 0.0
    return df_last5['MIN'].mean()

def PTS_last5(df_last5: pd.DataFrame) -> float:
    '''Calculate the average points scored over the last 5 games for a specific player.'''
    if df_last5.empty:
        return 0.0
    return df_last5['PTS'].mean()

def REB_last5(df_last5: pd.DataFrame) -> float:
    '''Calculate the average rebounds over the last 5 games for a specific player.'''
    if df_last5.empty:
        return 0.0
    return df_last5['REB'].mean()

def AST_last5(df_last5: pd.DataFrame) -> float:
    '''Calculate the average assists over the last 5 games for a specific player.'''
    if df_last5.empty:
        return 0.0
    return df_last5['AST'].mean()

def USAGE_last5(df_last5: pd.DataFrame) -> float:
    '''Calculate the average usage rate over the last 5 games for a specific player.'''
    if df_last5.empty:
        return 0.0
    return (df_last5['FGA'].mean() + df_last5['FTA'].mean())

def IS_HOME(df: pd.DataFrame) -> int:
    '''Determine if the player was playing a home game.'''
    try:
        matchup = df['MATCHUP'].values[0]
        return 1 if ' vs. ' in matchup else 0
    except IndexError:
        return 0
    
def DAYS_REST(df_last2: pd.DataFrame) -> int:
    '''Calculate the number of days of rest before the current game for a specific player.'''
    try:
        game_dates = pd.to_datetime(df_last2['GAME_DATE'], errors='coerce').dropna().sort_values(ascending=False).reset_index(drop=True)
        if len(game_dates) < 2:
            return 0
        days_rest = game_dates.iloc[0] - game_dates.iloc[1]
        return int(days_rest.days)
    except IndexError:
        return 0
    
def PLUS_MINUS_last5(df_last5: pd.DataFrame) -> float:
    '''Calculate the average plus-minus over the last 5 games for a specific player.'''
    if df_last5.empty:
        return 0.0
    return df_last5['PLUS_MINUS'].mean()       

def FG_PCT_last5(df_last5: pd.DataFrame) -> float:
    '''Calculate the average field goal percentage over the last 5 games for a specific player.'''   
    if df_last5.empty:
        return 0.0
    return df_last5['FG_PCT'].mean()

def GET_LAST_N_GAMES(df: pd.DataFrame, n: int, player_id: int, game_id: str):
    '''Retrieve the last N games for a specific player before a given game ID.'''
    player_games = df[df['PLAYER_ID'] == player_id]
    player_games = player_games.sort_values(by='GAME_DATE', ascending=False)
    game_idx = player_games[player_games['GAME_ID'] == game_id].index
    player_games = player_games[player_games.index > game_idx[0]]
    return player_games.head(n)

def main():
    df = pd.read_parquet('../parquet/player_game_logs_2025-26_FINAL.parquet')
    
    for row in df.itertuples():
        print(f"Processing row {row.Index + 1}")
        # add custom features for each game
        player_id = row.PLAYER_ID
        game_id = row.GAME_ID
        df_last5 = GET_LAST_N_GAMES(df, 5, player_id, game_id)
        df_last2 = GET_LAST_N_GAMES(df, 2, player_id, game_id)
        df.at[row.Index, 'MIN_last5'] = MIN_last5(df_last5)
        df.at[row.Index, 'PTS_last5'] = PTS_last5(df_last5)
        df.at[row.Index, 'REB_last5'] = REB_last5(df_last5)
        df.at[row.Index, 'AST_last5'] = AST_last5(df_last5)
        df.at[row.Index, 'USAGE_last5'] = USAGE_last5(df_last5)
        df.at[row.Index, 'IS_HOME'] = IS_HOME(df[df['GAME_ID'] == game_id])
        df.at[row.Index, 'DAYS_REST'] = DAYS_REST(df_last2)
        df.at[row.Index, 'PLUS_MINUS_last5'] = PLUS_MINUS_last5(df_last5)
        df.at[row.Index, 'FG_PCT_last5'] = FG_PCT_last5(df_last5)
        
    df.to_parquet('../parquet/player_game_logs_with_features_2025-26_FINAL.parquet', index=False)
    print(df)
        
if __name__ == "__main__":
    main()