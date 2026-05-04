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

> Connects to `csci403` on `ada.mines.edu` under schema `group120836`. Make sure you're on the Mines network (or VPN).

**Train the model**

```bash
python multi_model.py
```

Trains a multi-output neural net and reports MAE for each target across the test set.

---

## 2. Website

**Run the Flask app**

```bash
python website/app.py
```

You'll be prompted for your `ada.mines.edu` credentials in the terminal. Then open:

```
http://localhost:6768
```

Browse player radar charts, rolling stat trends, and team offensive/defensive rating history.
