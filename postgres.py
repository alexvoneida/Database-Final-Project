import pandas as pd

df1 = pd.read_parquet("parquet/final_database_2023-24.parquet")
df2 = pd.read_parquet("parquet/final_database_2024-25.parquet")
df3 = pd.read_parquet("parquet/final_database_2025-26.parquet")
df = pd.concat([df1, df2, df3], ignore_index=True)

# --- players ---
players = (
    df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]]
    .drop_duplicates("PLAYER_ID")
    .rename(columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "player_name",
                     "TEAM_ABBREVIATION": "team_abbreviation"})
)

# --- games ---
# Parse home/away from MATCHUP (format: "GSW vs. LAL" or "GSW @ LAL")
def parse_matchup(row):
    parts = row["MATCHUP"].replace("@", "vs.").split("vs.")
    team = row["TEAM_ABBREVIATION"].strip()
    opp  = row["OPP_ABBREVIATION"].strip()
    home = team if row["IS_HOME"] else opp
    away = opp  if row["IS_HOME"] else team
    return home, away

df[["home_team", "away_team"]] = df.apply(parse_matchup, axis=1, result_type="expand")

games = (
    df[["GAME_ID", "GAME_DATE", "MATCHUP", "home_team", "away_team"]]
    .drop_duplicates("GAME_ID")
    .rename(columns={"GAME_ID": "game_id", "GAME_DATE": "game_date", "MATCHUP": "matchup"})
)

def infer_season(date_str):
    year = pd.to_datetime(date_str).year
    month = pd.to_datetime(date_str).month
    start = year if month >= 10 else year - 1
    return f"{start}-{str(start + 1)[-2:]}"  # e.g. "2023-24"

games["season_year"] = games["game_date"].apply(infer_season)

# --- player_game_stats ---
player_game_stats = df[[
    "PLAYER_ID", "GAME_ID", "PTS", "AST", "REB", "MIN",
    "FGA", "FTA", "FG_PCT", "PLUS_MINUS", "IS_HOME", "DAYS_REST"
]].rename(columns=str.lower).rename(columns={"player_id": "player_id", "game_id": "game_id"})

# --- player_rolling_stats ---
player_rolling_stats = df[[
    "PLAYER_ID", "GAME_ID",
    "MIN_last5", "PTS_last5", "REB_last5", "AST_last5",
    "USAGE_last5", "PLUS_MINUS_last5", "FG_PCT_last5"
]].rename(columns=str.lower)

# --- team_game_stats ---
team_game_stats = df[[
    "GAME_ID", "TEAM_ABBREVIATION", "OPP_ABBREVIATION",
    "offensiveRating_last5", "defensiveRating_last5", "pace_last5",
    "OPP_offensiveRating_last5", "OPP_defensiveRating_last5", "OPP_pace_last5"
]].drop_duplicates(["GAME_ID", "TEAM_ABBREVIATION"]).rename(columns={
    "GAME_ID": "game_id",
    "TEAM_ABBREVIATION": "team_abbreviation",
    "OPP_ABBREVIATION": "opp_abbreviation",
    "offensiveRating_last5": "offensive_rating_last5",
    "defensiveRating_last5": "defensive_rating_last5",
    "pace_last5": "pace_last5",
    "OPP_offensiveRating_last5": "opp_offensive_rating_last5",
    "OPP_defensiveRating_last5": "opp_defensive_rating_last5",
    "OPP_pace_last5": "opp_pace_last5",
})

# --- Export ---
players.to_csv("postgres/players.csv", index=False)
games.to_csv("postgres/games.csv", index=False)
player_game_stats.to_csv("postgres/player_game_stats.csv", index=False)
player_rolling_stats.to_csv("postgres/player_rolling_stats.csv", index=False)
team_game_stats.to_csv("postgres/team_game_stats.csv", index=False)