from flask import Flask, render_template, request
import pg8000
import getpass
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multi_model import NBAMultiOutputModel

app = Flask(__name__)

login = input('Login username: ')
secret = getpass.getpass()

_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'multi_output_model.pth')
try:
    prediction_model = torch.load(_MODEL_PATH, map_location='cpu', weights_only=False)
    prediction_model.eval()
except Exception as e:
    print(f"Warning: could not load prediction model: {e}")
    prediction_model = None

def get_db_connection():
    conn = pg8000.connect(
        user=login,
        password=secret,
        database='csci403',
        host='ada.mines.edu'
    )
    cursor = conn.cursor()
    cursor.execute("SET search_path TO group120836")
    return conn

# --- ROUTES ---

@app.route('/')
def index():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT team_abbreviation FROM players ORDER BY team_abbreviation")
    teams = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT player_id, player_name FROM players ORDER BY player_name")
    players = cursor.fetchall()

    conn.close()
    return render_template('index.html', teams=teams, players=players)

@app.route('/player/<int:player_id>')
def player_stats(player_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT player_name, team_abbreviation
            FROM group120836.players
            WHERE player_id = %s
        """, (player_id,))
        player_info = cursor.fetchone()

        if not player_info:
            return f"Player ID {player_id} not found.", 404

        cursor.execute("""
            SELECT pgs.game_id, g.game_date, g.matchup
            FROM group120836.player_game_stats pgs
            JOIN group120836.games g ON pgs.game_id = g.game_id
            WHERE pgs.player_id = %s
            ORDER BY g.game_date DESC
            LIMIT 50
        """, (player_id,))
        games = cursor.fetchall()

        conn.close()
        return render_template('player.html', info=player_info, games=games, player_id=player_id)

    except Exception as e:
        if conn:
            conn.close()
        return f"Database Error: {str(e)}", 500

@app.route('/player/<int:player_id>/game/<string:game_id>')
def game_stats(player_id, game_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT player_name, team_abbreviation
            FROM group120836.players
            WHERE player_id = %s
        """, (player_id,))
        player_info = cursor.fetchone()
        if not player_info:
            return f"Player ID {player_id} not found.", 404

        cursor.execute("""
            SELECT game_date, matchup
            FROM group120836.games
            WHERE game_id = %s
        """, (game_id,))
        game_info = cursor.fetchone()

        cursor.execute("""
            SELECT pts_last5, reb_last5, ast_last5, usage_last5, fg_pct_last5,
                   min_last5, plus_minus_last5
            FROM group120836.player_rolling_stats
            WHERE player_id = %s AND game_id = %s
        """, (player_id, game_id))
        rolling = cursor.fetchone()

        cursor.execute("""
            SELECT pgs.pts, g.matchup
            FROM group120836.player_game_stats pgs
            JOIN group120836.games g ON pgs.game_id = g.game_id
            WHERE pgs.player_id = %s
              AND g.game_date <= (SELECT game_date FROM group120836.games WHERE game_id = %s)
            ORDER BY g.game_date DESC LIMIT 10
        """, (player_id, game_id))
        raw_trend = cursor.fetchall()
        pts_history = [row[0] for row in reversed(raw_trend)]
        trend_labels = [row[1] for row in reversed(raw_trend)]

        cursor.execute("""
            SELECT pts, ast, reb, min, fg_pct, plus_minus, is_home, days_rest
            FROM group120836.player_game_stats
            WHERE player_id = %s AND game_id = %s
        """, (player_id, game_id))
        game_pgs = cursor.fetchone()

        team_abbr = player_info[1]
        cursor.execute("""
            SELECT offensive_rating_last5, defensive_rating_last5, pace_last5,
                   opp_offensive_rating_last5, opp_defensive_rating_last5, opp_pace_last5
            FROM group120836.team_game_stats
            WHERE game_id = %s AND team_abbreviation = %s
        """, (game_id, team_abbr))
        team = cursor.fetchone()

        conn.close()

        # Per-stat ceilings used only for radar normalization (0–100 scale per axis).
        # pts, reb, ast, usage, fg_pct (raw 0-1)
        _RADAR_MAXES = [40.0, 15.0, 12.0, 30.0, 1.0]

        stats = None
        radar_data = [0, 0, 0, 0, 0]
        if rolling:
            stats = (rolling[0], rolling[1], rolling[2], rolling[3], round(float(rolling[4]) * 100, 1))
            raw = [float(rolling[0]), float(rolling[1]), float(rolling[2]), float(rolling[3]), float(rolling[4])]
            radar_data = [round(min(v / m * 100, 100), 1) for v, m in zip(raw, _RADAR_MAXES)]

        prediction = None
        actual = None
        if game_pgs:
            actual = {
                'pts': game_pgs[0],
                'ast': game_pgs[1],
                'reb': game_pgs[2],
                'min': game_pgs[3],
                'fg_pct': round(float(game_pgs[4]) * 100, 1),
                'plus_minus': game_pgs[5],
            }

        if prediction_model and rolling and team and game_pgs:
            feat = [
                float(rolling[5]),        # min_last5
                float(rolling[0]),        # pts_last5
                float(rolling[1]),        # reb_last5
                float(rolling[2]),        # ast_last5
                float(rolling[4]),        # fg_pct_last5 (raw 0-1)
                float(rolling[3]),        # usage_last5
                float(int(game_pgs[6])),  # is_home
                float(game_pgs[7]),       # days_rest
                float(rolling[6]),        # plus_minus_last5
                float(team[0]),           # offensive_rating_last5
                float(team[1]),           # defensive_rating_last5
                float(team[2]),           # pace_last5
                float(team[3]),           # opp_offensive_rating_last5
                float(team[4]),           # opp_defensive_rating_last5
                float(team[5]),           # opp_pace_last5
            ]
            x = torch.tensor([feat], dtype=torch.float32)
            with torch.no_grad():
                pred = prediction_model(x).numpy()[0]
            prediction = {
                'pts': round(float(pred[0]), 1),
                'reb': round(float(pred[1]), 1),
                'ast': round(float(pred[2]), 1),
                'fg_pct': round(float(pred[3]) * 100, 1),
            }

        return render_template('game.html',
                               info=player_info,
                               game_info=game_info,
                               stats=stats,
                               radar_data=radar_data,
                               pts_history=pts_history,
                               labels=trend_labels,
                               actual=actual,
                               prediction=prediction,
                               player_id=player_id)

    except Exception as e:
        if conn:
            conn.close()
        return f"Database Error: {str(e)}", 500

@app.route('/team/<string:team_abbr>')
def team_stats(team_abbr):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Matching columns exactly to IMG_0898[1].jpg
    query = """
        SELECT offensive_rating_last5, defensive_rating_last5, pace_last5 
        FROM group120836.team_game_stats 
        WHERE team_abbreviation = %s 
        ORDER BY id DESC LIMIT 15
    """
    
    try:
        cursor.execute(query, (team_abbr,))
        rows = cursor.fetchall()
        conn.close() # Close connection after fetching
    except Exception as e:
        if conn: conn.close()
        return f"Database Error: {str(e)}" # Returns a string (valid response)

    # Check if we actually got data back
    if not rows:
        return f"No data found for team: {team_abbr}"

    # Prepare data for Chart.js
    list(rows).reverse()
    off_ratings = [row[0] for row in rows]
    def_ratings = [row[1] for row in rows]
    labels = [f"Game {i+1}" for i in range(len(rows))]

    # CRITICAL: This return statement must be present and outside the try/except/if blocks
    return render_template('team.html', 
                           team_abbr=team_abbr, 
                           off_ratings=off_ratings, 
                           def_ratings=def_ratings, 
                           labels=labels)



if __name__ == '__main__':
    # Setting debug=True helps see errors in the browser
    app.run(host='0.0.0.0', port=6768, debug=True, use_reloader=False)