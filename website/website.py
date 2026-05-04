from flask import Flask, render_template, request
import pg8000
import getpass

app = Flask(__name__)

# --- DATABASE CONNECTION SETUP ---
# This will prompt you in the terminal when you start the server
login = input('Login username: ')
secret = getpass.getpass()

def get_db_connection():
    conn = pg8000.connect(
        user=login,
        password=secret,
        database='csci403',
        host='ada.mines.edu'
    )
    cursor = conn.cursor()
    # Ensure we are looking at your specific schema
    cursor.execute("SET search_path TO group120836")
    return conn

# --- ROUTES ---

@app.route('/')
def index():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get unique teams for the dropdown
    cursor.execute("SELECT DISTINCT team_abbreviation FROM players ORDER BY team_abbreviation")
    teams = [row[0] for row in cursor.fetchall()]
    
    # Get players for the dropdown
    cursor.execute("SELECT player_id, player_name FROM players ORDER BY player_name")
    players = cursor.fetchall()
    
    conn.close()
    return render_template('index.html', teams=teams, players=players)

@app.route('/player/<int:player_id>')
def player_stats(player_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # 1. Fetch Player Name and Team (Basic Info)
        # Table: players (player_id, player_name, team_abbreviation)
        cursor.execute("""
            SELECT player_name, team_abbreviation 
            FROM group120836.players 
            WHERE player_id = %s
        """, (player_id,))
        player_info = cursor.fetchone()

        if not player_info:
            return f"Player ID {player_id} not found.", 404

        # 2. Fetch Latest Rolling Stats (for Radar Chart and Quick Stats Cards)
        # Order must match the Radar labels: Points, Rebounds, Assists, Usage, FG%
        # Table: player_rolling_stats
        cursor.execute("""
            SELECT pts_last5, reb_last5, ast_last5, usage_last5, (fg_pct_last5 * 100)
            FROM group120836.player_rolling_stats 
            WHERE player_id = %s 
            ORDER BY id DESC LIMIT 1
        """, (player_id,))
        stats = cursor.fetchone()

        # 3. Fetch Historical Trend (Last 10 games for Line Chart)
        # Table: player_rolling_stats
        cursor.execute("""
            SELECT pts_last5 
            FROM group120836.player_rolling_stats 
            WHERE player_id = %s 
            ORDER BY id DESC LIMIT 10
        """, (player_id,))
        
        raw_trend = cursor.fetchall()
        # Reverse data so it reads left-to-right (oldest to newest)
        pts_history = [row[0] for row in reversed(raw_trend)]
        labels = [f"Game -{i}" for i in range(len(pts_history), 0, -1)]

        conn.close()

        # Pass all data to the template
        return render_template('player.html', 
                               info=player_info, 
                               stats=stats, 
                               radar_data=list(stats) if stats else [0,0,0,0,0],
                               pts_history=pts_history,
                               labels=labels)

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
    app.run(host='0.0.0.0', port=6768, debug=True)