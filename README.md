# NBA Stat Predictor

A PyTorch neural network that predicts **PTS, REB, AST, and FG%** for NBA players, backed by a PostgreSQL database and a Flask visualization dashboard.

---

## 1. Model & Database

**Set up your environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Load data into PostgreSQL**

```bash
python postgres.py
```

> Connects to `csci403` on `ada.mines.edu` under schema `group120836`. Make sure you're on the Mines network (or VPN). This step is only needed to rebuild the
> original Postgres tables — the website itself now runs off the local snapshot
> described below.

**Train the model**

```bash
python multi_model.py
```

Trains a multi-output neural net and reports MAE for each target across the test set.

---

## 2. Website

**Build the local database**

The app reads a SQLite snapshot built from the CSV exports in `postgres/`, so no
connection to `ada.mines.edu` is required.

```bash
python build_local_db.py
```

**Run the Flask app**

```bash
python website/app.py
```

Then open:

```
http://localhost:6768
```

Browse player radar charts, rolling stat trends, and team offensive/defensive rating history.
