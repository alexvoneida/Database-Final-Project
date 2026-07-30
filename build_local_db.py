"""Builds local.db (SQLite) from the CSV exports in postgres/ so the website can
run without access to the Postgres server on ada.mines.edu."""

import csv
import os
import sqlite3

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(REPO_ROOT, 'postgres')
DB_PATH = os.path.join(REPO_ROOT, 'local.db')

TABLES = {
    'players': """
        player_id INTEGER PRIMARY KEY,
        player_name TEXT,
        team_abbreviation TEXT
    """,
    'games': """
        game_id TEXT PRIMARY KEY,
        game_date TEXT,
        matchup TEXT,
        home_team TEXT,
        away_team TEXT,
        season_year TEXT
    """,
    'player_game_stats': """
        player_id INTEGER,
        game_id TEXT,
        pts INTEGER,
        ast INTEGER,
        reb INTEGER,
        min REAL,
        fga INTEGER,
        fta INTEGER,
        fg_pct REAL,
        plus_minus REAL,
        is_home INTEGER,
        days_rest REAL
    """,
    'player_rolling_stats': """
        player_id INTEGER,
        game_id TEXT,
        min_last5 REAL,
        pts_last5 REAL,
        reb_last5 REAL,
        ast_last5 REAL,
        usage_last5 REAL,
        plus_minus_last5 REAL,
        fg_pct_last5 REAL
    """,
    'team_game_stats': """
        game_id TEXT,
        team_abbreviation TEXT,
        opp_abbreviation TEXT,
        offensive_rating_last5 REAL,
        defensive_rating_last5 REAL,
        pace_last5 REAL,
        opp_offensive_rating_last5 REAL,
        opp_defensive_rating_last5 REAL,
        opp_pace_last5 REAL
    """,
}

INDEXES = [
    "CREATE INDEX idx_pgs_player ON player_game_stats(player_id, game_id)",
    "CREATE INDEX idx_prs_player ON player_rolling_stats(player_id, game_id)",
    "CREATE INDEX idx_tgs_team ON team_game_stats(team_abbreviation, game_id)",
]


def load_table(conn, table, column_definitions):
    path = os.path.join(CSV_DIR, f'{table}.csv')
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[value if value != '' else None for value in row] for row in reader]

    conn.execute(f'DROP TABLE IF EXISTS {table}')
    conn.execute(f'CREATE TABLE {table} ({column_definitions})')
    placeholders = ','.join('?' * len(header))
    columns = ','.join(f'"{name}"' for name in header)
    conn.executemany(f'INSERT INTO {table} ({columns}) VALUES ({placeholders})', rows)
    print(f'{table}: {len(rows)} rows')


def main():
    conn = sqlite3.connect(DB_PATH)
    for table, column_definitions in TABLES.items():
        load_table(conn, table, column_definitions)
    for statement in INDEXES:
        conn.execute(statement)
    conn.commit()
    conn.close()
    print(f'\nWrote {DB_PATH}')


if __name__ == '__main__':
    main()
