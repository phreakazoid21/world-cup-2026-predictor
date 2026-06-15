from __future__ import annotations
"""
FIFA World Cup 2026 – Match Prediction Model
=============================================
A complete, production-ready ML pipeline for predicting match outcomes.

Architecture:
  1. Data layer   – historical World Cup + international match data
  2. Feature eng  – Elo ratings, recent form, head-to-head, tournament context
  3. Model layer  – XGBoost classifier (home win / draw / away win)
  4. Prediction   – probabilities + expected goals for any upcoming match
  5. Visualisation– charts ready to drop into your YouTube thumbnails/videos

Usage:
  pip install -r requirements.txt
  python predictor.py                    # demo mode: predict several WC26 matches
  python predictor.py --match "Brazil" "France"
  python predictor.py --simulate         # full tournament Monte Carlo simulation
"""

import argparse
import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

DATA_DIR  = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
VIS_DIR   = Path(__file__).parent / "visuals"
for d in [DATA_DIR, MODEL_DIR, VIS_DIR]:
    d.mkdir(exist_ok=True)

# Free dataset: international results 1872–present (Kaggle / GitHub mirror)
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# Live FIFA World Cup 2026 results + schedule (openfootball, public domain, no key)
WC2026_JSON_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
)

# Co-hosts of WC2026 — given home advantage when listed as the home side
HOST_NATIONS = {"USA", "Canada", "Mexico"}

# Canonical team names: the same nation is spelled differently across the
# historical dataset (martj42), the openfootball feed, and our schedule below.
# Everything is mapped to ONE spelling so Elo ratings line up end-to-end.
TEAM_NAME_MAP = {
    "Czech Republic":         "Czechia",
    "Korea Republic":         "South Korea",
    "Korea DPR":              "North Korea",
    "United States":          "USA",
    "Bosnia & Herzegovina":   "Bosnia and Herzegovina",
    "Bosnia-Herzegovina":     "Bosnia and Herzegovina",
    "Curaçao":                "Curacao",
    "Côte d'Ivoire":          "Ivory Coast",
    "Cote d'Ivoire":          "Ivory Coast",
    "IR Iran":                "Iran",
    "Türkiye":                "Turkey",
    "Turkiye":                "Turkey",
    "Cabo Verde":             "Cape Verde",
    "China PR":               "China",
}


def normalize_team(name: str) -> str:
    """Map differing data-source spellings of a nation to one canonical name."""
    n = str(name).strip()
    return TEAM_NAME_MAP.get(n, n)

# FIFA World Cup 2026 group stage schedule (openfootball public domain data)
WC2026_MATCHES = [
    # ── GROUP A: Mexico · South Africa · South Korea · Czechia ──
    ("Mexico",        "South Africa",          "2026-06-11"),
    ("South Korea",   "Czechia",               "2026-06-11"),
    ("Mexico",        "Czechia",               "2026-06-17"),
    ("South Korea",   "South Africa",          "2026-06-17"),
    ("Mexico",        "South Korea",           "2026-06-22"),
    ("Czechia",       "South Africa",          "2026-06-22"),

    # ── GROUP B: Canada · Switzerland · Qatar · Bosnia and Herzegovina ──
    ("Canada",                    "Bosnia and Herzegovina", "2026-06-12"),
    ("Qatar",                     "Switzerland",            "2026-06-13"),
    ("Canada",                    "Qatar",                  "2026-06-18"),
    ("Bosnia and Herzegovina",    "Switzerland",            "2026-06-18"),
    ("Canada",                    "Switzerland",            "2026-06-23"),
    ("Bosnia and Herzegovina",    "Qatar",                  "2026-06-23"),

    # ── GROUP C: Brazil · Morocco · Haiti · Scotland ──
    ("Brazil",        "Morocco",               "2026-06-13"),
    ("Haiti",         "Scotland",              "2026-06-13"),
    ("Brazil",        "Scotland",              "2026-06-18"),
    ("Morocco",       "Haiti",                 "2026-06-19"),
    ("Brazil",        "Haiti",                 "2026-06-23"),
    ("Scotland",      "Morocco",               "2026-06-24"),

    # ── GROUP D: USA · Paraguay · Australia · Turkey ──
    ("USA",           "Paraguay",              "2026-06-12"),
    ("Australia",     "Turkey",                "2026-06-13"),
    ("USA",           "Australia",             "2026-06-19"),
    ("Paraguay",      "Turkey",                "2026-06-19"),
    ("USA",           "Turkey",                "2026-06-25"),
    ("Paraguay",      "Australia",             "2026-06-25"),

    # ── GROUP E: Germany · Curacao · Ivory Coast · Ecuador ──
    ("Germany",       "Curacao",               "2026-06-14"),
    ("Ivory Coast",   "Ecuador",               "2026-06-14"),
    ("Germany",       "Ivory Coast",           "2026-06-20"),
    ("Ecuador",       "Curacao",               "2026-06-20"),
    ("Germany",       "Ecuador",               "2026-06-25"),
    ("Ivory Coast",   "Curacao",               "2026-06-25"),

    # ── GROUP F: Netherlands · Japan · Sweden · Tunisia ──
    ("Netherlands",   "Japan",                 "2026-06-14"),
    ("Sweden",        "Tunisia",               "2026-06-14"),
    ("Netherlands",   "Sweden",                "2026-06-20"),
    ("Tunisia",       "Japan",                 "2026-06-20"),
    ("Japan",         "Sweden",                "2026-06-25"),
    ("Tunisia",       "Netherlands",           "2026-06-25"),

    # ── GROUP G: Belgium · Egypt · Iran · New Zealand ──
    ("Belgium",       "Egypt",                 "2026-06-15"),
    ("Iran",          "New Zealand",           "2026-06-15"),
    ("Belgium",       "Iran",                  "2026-06-21"),
    ("New Zealand",   "Egypt",                 "2026-06-21"),
    ("Egypt",         "Iran",                  "2026-06-26"),
    ("New Zealand",   "Belgium",               "2026-06-26"),

    # ── GROUP H: Spain · Cape Verde · Saudi Arabia · Uruguay ──
    ("Spain",         "Cape Verde",            "2026-06-15"),
    ("Saudi Arabia",  "Uruguay",               "2026-06-15"),
    ("Spain",         "Saudi Arabia",          "2026-06-21"),
    ("Uruguay",       "Cape Verde",            "2026-06-21"),
    ("Spain",         "Uruguay",               "2026-06-26"),
    ("Cape Verde",    "Saudi Arabia",          "2026-06-26"),

    # ── GROUP I: France · Senegal · Iraq · Norway ──
    ("France",        "Senegal",               "2026-06-16"),
    ("Iraq",          "Norway",                "2026-06-16"),
    ("France",        "Iraq",                  "2026-06-22"),
    ("Norway",        "Senegal",               "2026-06-22"),
    ("France",        "Norway",                "2026-06-26"),
    ("Senegal",       "Iraq",                  "2026-06-26"),

    # ── GROUP J: Argentina · Algeria · Austria · Jordan ──
    ("Argentina",     "Algeria",               "2026-06-16"),
    ("Austria",       "Jordan",                "2026-06-16"),
    ("Argentina",     "Jordan",                "2026-06-22"),
    ("Algeria",       "Austria",               "2026-06-22"),
    ("Argentina",     "Austria",               "2026-06-26"),
    ("Jordan",        "Algeria",               "2026-06-26"),

    # ── GROUP K: Portugal · DR Congo · Uzbekistan · Colombia ──
    ("Portugal",      "DR Congo",              "2026-06-17"),
    ("Uzbekistan",    "Colombia",              "2026-06-17"),
    ("Portugal",      "Uzbekistan",            "2026-06-23"),
    ("Colombia",      "DR Congo",              "2026-06-23"),
    ("Portugal",      "Colombia",              "2026-06-27"),
    ("DR Congo",      "Uzbekistan",            "2026-06-27"),

    # ── GROUP L: England · Croatia · Ghana · Panama ──
    ("England",       "Croatia",               "2026-06-17"),
    ("Ghana",         "Panama",                "2026-06-17"),
    ("England",       "Ghana",                 "2026-06-23"),
    ("Croatia",       "Panama",                "2026-06-23"),
    ("England",       "Panama",                "2026-06-27"),
    ("Croatia",       "Ghana",                 "2026-06-27"),
]

# The 48 nations actually at WC2026 (used to filter out knockout placeholders
# like "1A" / "W73" that openfootball uses before the bracket is decided).
WC2026_TEAMS = {normalize_team(t) for m in WC2026_MATCHES for t in (m[0], m[1])}


# ──────────────────────────────────────────────
# STEP 1 – DATA LOADING
# ──────────────────────────────────────────────

def load_results(max_rows: int = 50_000, include_wc2026: bool = True) -> pd.DataFrame:
    """
    Load historical international match results (martj42 dataset), then merge in
    the latest *played* World Cup 2026 matches from the openfootball feed so the
    Elo ratings and model reflect games that have already happened.

    Falls back to synthetic data if the network is unavailable
    (useful for offline testing / sandboxed environments).
    """
    cache = DATA_DIR / "results.csv"
    if cache.exists():
        print("📂  Loading cached match history …")
        df = pd.read_csv(cache)
    else:
        print("📡  Downloading international results dataset …")
        try:
            df = pd.read_csv(RESULTS_URL)
            df.to_csv(cache, index=False)
            print(f"    Saved {len(df):,} matches to cache.")
        except Exception as e:
            print(f"    Network unavailable ({e}). Generating synthetic dataset …")
            df = _synthetic_results(n=max_rows)
            df.to_csv(cache, index=False)

    df["date"] = pd.to_datetime(df["date"])
    # Keep only matches from 1990 onward (modern football)
    df = df[df["date"] >= "1990-01-01"].copy()
    # Drop rows with missing scores (NaN) — real dataset has some future/cancelled matches
    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    dropped = before - len(df)
    if dropped > 0:
        print(f"    Dropped {dropped} rows with missing scores.")

    # Unify team-name spellings so they match the schedule + openfootball feed
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    print(f"    {len(df):,} historical matches loaded (1990–present).")

    # Merge in the latest played WC2026 results from openfootball
    if include_wc2026:
        wc = load_wc2026_results()
        if not wc.empty:
            df = pd.concat([df, wc], ignore_index=True)
            # openfootball is authoritative for WC2026 → drop any stale duplicate
            df = df.drop_duplicates(
                subset=["date", "home_team", "away_team"], keep="last"
            )

    df = df.sort_values("date").reset_index(drop=True)
    return df


def _synthetic_results(n: int = 50_000) -> pd.DataFrame:
    """
    Generates realistic-looking synthetic match data so the full pipeline
    runs even in air-gapped / sandboxed environments.
    """
    rng = np.random.default_rng(42)
    teams = [
        "Brazil","Germany","France","Argentina","Spain","England","Italy",
        "Netherlands","Portugal","Belgium","Croatia","Uruguay","Mexico",
        "USA","Japan","South Korea","Colombia","Chile","Denmark","Sweden",
        "Switzerland","Poland","Senegal","Nigeria","Egypt","Morocco",
        "Saudi Arabia","Iran","Australia","Canada","South Africa","Panama",
    ]
    strengths = {t: rng.uniform(0.3, 1.0) for t in teams}
    # Make big teams stronger
    for t, v in [("Brazil",1.0),("Germany",0.95),("France",0.95),
                 ("Argentina",0.93),("Spain",0.90),("England",0.87)]:
        strengths[t] = v

    rows = []
    dates = pd.date_range("1990-01-01", "2026-05-01", periods=n)
    tournaments = ["FIFA World Cup","Friendly","Copa America","Euro","Gold Cup","Qualifier"]
    for i in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        lam_h = strengths[h] * 1.5 + 0.2  # slight home advantage
        lam_a = strengths[a] * 1.3
        gh = int(rng.poisson(lam_h))
        ga = int(rng.poisson(lam_a))
        rows.append({
            "date":       dates[i].strftime("%Y-%m-%d"),
            "home_team":  h,
            "away_team":  a,
            "home_score": gh,
            "away_score": ga,
            "tournament": rng.choice(tournaments, p=[0.08,0.5,0.08,0.08,0.08,0.18]),
            "neutral":    rng.random() > 0.6,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# STEP 1b – LIVE WORLD CUP 2026 DATA (openfootball)
# ──────────────────────────────────────────────

def _fetch_wc2026() -> dict:
    """
    Fetch the latest openfootball WC2026 feed (104 matches, ~100 KB).
    Always tries the network first so results stay current, falling back to a
    local cache when offline.
    """
    cache = DATA_DIR / "worldcup2026.json"
    try:
        resp = requests.get(WC2026_JSON_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        cache.write_text(json.dumps(data))
        return data
    except Exception as e:
        if cache.exists():
            print(f"    ⚠️  Network unavailable ({e}); using cached WC2026 feed.")
            return json.loads(cache.read_text())
        print(f"    ⚠️  Could not fetch WC2026 feed ({e}).")
        return {"matches": []}


def _is_real_team(name: str) -> bool:
    """False for bracket placeholders like '1A', '2B', '3A/B/C/D/F', 'W73', 'L101'."""
    n = str(name).strip()
    return bool(n) and not re.match(r"^(\d|[WL]\d|.+/)", n)


def load_wc2026_results() -> pd.DataFrame:
    """
    Latest *played* WC2026 matches, reshaped to the historical training schema:
    date / home_team / away_team / home_score / away_score / tournament / neutral.
    """
    rows = []
    for m in _fetch_wc2026().get("matches", []):
        ft = (m.get("score") or {}).get("ft")
        if not ft or len(ft) != 2:
            continue  # not played yet
        home = normalize_team(m.get("team1", ""))
        away = normalize_team(m.get("team2", ""))
        if not (_is_real_team(home) and _is_real_team(away)):
            continue
        rows.append({
            "date":       m.get("date"),
            "home_team":  home,
            "away_team":  away,
            "home_score": int(ft[0]),
            "away_score": int(ft[1]),
            "tournament": "FIFA World Cup",
            # WC matches are at neutral venues except when a co-host plays at home
            "neutral":    home not in HOST_NATIONS,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        print(f"    🏟️  Merged {len(df)} played WC2026 matches from openfootball.")
    return df


def upcoming_fixtures() -> pd.DataFrame:
    """
    Scheduled WC2026 matches that have NOT been played yet and whose teams are
    already confirmed (knockout slots with placeholder teams are skipped).
    Returns date / home_team / away_team / round / group / ground / neutral.
    """
    rows = []
    for m in _fetch_wc2026().get("matches", []):
        if (m.get("score") or {}).get("ft"):
            continue  # already played
        home = normalize_team(m.get("team1", ""))
        away = normalize_team(m.get("team2", ""))
        if home not in WC2026_TEAMS or away not in WC2026_TEAMS:
            continue  # knockout matchup not decided yet
        rows.append({
            "date":      m.get("date"),
            "home_team": home,
            "away_team": away,
            "round":     m.get("round"),
            "group":     m.get("group"),
            "ground":    m.get("ground"),
            "neutral":   home not in HOST_NATIONS,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# STEP 2 – ELO RATING ENGINE
# ──────────────────────────────────────────────

class EloRatingSystem:
    """
    Dynamic Elo rating system for national teams.
    Ratings update after every match and weight tournament matches
    more heavily than friendlies.
    """

    # Tournament importance multipliers (k-factor weights)
    TOURNAMENT_WEIGHTS = {
        "FIFA World Cup":          60,
        "Copa America":            45,
        "UEFA Euro":               45,
        "Africa Cup of Nations":   45,
        "CONCACAF Gold Cup":       35,
        "Asian Cup":               35,
        "World Cup Qualification": 30,
        "Friendly":                20,
    }
    DEFAULT_K = 25
    INITIAL_ELO = 1500

    def __init__(self):
        self.ratings: dict[str, float] = defaultdict(lambda: self.INITIAL_ELO)
        self.history: dict[str, list] = defaultdict(list)

    def _k(self, tournament: str) -> float:
        for key, k in self.TOURNAMENT_WEIGHTS.items():
            if key.lower() in tournament.lower():
                return k
        return self.DEFAULT_K

    def expected(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, home: str, away: str, gh: int, ga: int,
               tournament: str, neutral: bool):
        ra, rb = self.ratings[home], self.ratings[away]
        # Home advantage: +100 Elo unless neutral venue
        if not neutral:
            ra += 100
        ea = self.expected(ra, rb)

        if gh > ga:
            sa = 1.0
        elif gh == ga:
            sa = 0.5
        else:
            sa = 0.0

        # Goal difference multiplier (margin of victory)
        gd = abs(gh - ga)
        gd_mult = math.log(max(gd, 1) + 1) + (1 if gd > 1 else 0)

        k = self._k(tournament) * gd_mult
        delta = k * (sa - ea)

        self.ratings[home] += delta
        self.ratings[away] -= delta
        self.history[home].append(self.ratings[home])
        self.history[away].append(self.ratings[away])

    def fit(self, df: pd.DataFrame) -> "EloRatingSystem":
        df_sorted = df.sort_values("date")
        for _, row in df_sorted.iterrows():
            self.update(
                row["home_team"], row["away_team"],
                int(row["home_score"]), int(row["away_score"]),
                str(row.get("tournament", "Friendly")),
                bool(row.get("neutral", False)),
            )
        return self

    def top_n(self, n: int = 20) -> pd.DataFrame:
        return (
            pd.DataFrame(self.ratings.items(), columns=["team", "elo"])
            .sort_values("elo", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )


# ──────────────────────────────────────────────
# STEP 3 – FEATURE ENGINEERING
# ──────────────────────────────────────────────

def build_features(df: pd.DataFrame, elo: EloRatingSystem) -> pd.DataFrame:
    """
    Build a feature matrix from historical match data.
    Each row = one match, with features computed at the time of the match.
    """
    print("⚙️   Engineering features …")

    # Rebuild Elo game-by-game to get snapshot ratings at match time
    ratings_snap: dict[str, float] = defaultdict(lambda: EloRatingSystem.INITIAL_ELO)
    history = []

    for _, row in df.sort_values("date").iterrows():
        home, away = row["home_team"], row["away_team"]
        gh, ga = int(row["home_score"]), int(row["away_score"])
        neutral = bool(row.get("neutral", False))
        tournament = str(row.get("tournament", "Friendly"))

        elo_h = ratings_snap[home]
        elo_a = ratings_snap[away]
        elo_diff = elo_h - elo_a + (0 if neutral else 100)

        # Outcome label: 0=away win, 1=draw, 2=home win
        if gh > ga:
            outcome = 2
        elif gh == ga:
            outcome = 1
        else:
            outcome = 0

        history.append({
            "date":           row["date"],
            "home_team":      home,
            "away_team":      away,
            "home_score":     gh,
            "away_score":     ga,
            "elo_home":       elo_h,
            "elo_away":       elo_a,
            "elo_diff":       elo_diff,
            "neutral":        int(neutral),
            "is_wc":          int("world cup" in tournament.lower()),
            "outcome":        outcome,
        })

        # Update snapshot ratings
        elo_obj = EloRatingSystem()
        elo_obj.ratings = dict(ratings_snap)
        elo_obj.update(home, away, gh, ga, tournament, neutral)
        ratings_snap[home] = elo_obj.ratings[home]
        ratings_snap[away] = elo_obj.ratings[away]

    feat_df = pd.DataFrame(history)

    # Rolling form: avg goals scored/conceded in last 5 matches (per team)
    feat_df = _add_rolling_form(feat_df)

    # Head-to-head record (last 10 meetings)
    feat_df = _add_h2h(feat_df)

    print(f"    Feature matrix: {feat_df.shape}")
    return feat_df


def _add_rolling_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling avg goals scored and conceded for each team."""
    scored_h, conceded_h = {}, {}
    scored_a, conceded_a = {}, {}

    rows = []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        gh, ga = row["home_score"], row["away_score"]

        h_scored   = np.mean(scored_h.get(h, [1.5])[-window:])
        h_conceded = np.mean(conceded_h.get(h, [1.2])[-window:])
        a_scored   = np.mean(scored_a.get(a, [1.5])[-window:])
        a_conceded = np.mean(conceded_a.get(a, [1.2])[-window:])

        row = row.copy()
        row["home_form_scored"]   = h_scored
        row["home_form_conceded"] = h_conceded
        row["away_form_scored"]   = a_scored
        row["away_form_conceded"] = a_conceded
        row["form_diff"]          = (h_scored - h_conceded) - (a_scored - a_conceded)
        rows.append(row)

        scored_h.setdefault(h, []).append(gh)
        conceded_h.setdefault(h, []).append(ga)
        scored_a.setdefault(a, []).append(ga)
        conceded_a.setdefault(a, []).append(gh)

    return pd.DataFrame(rows)


def _add_h2h(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add head-to-head win rate for home team vs away team."""
    h2h: dict[tuple, list] = defaultdict(list)
    h2h_rates = []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        key  = tuple(sorted([h, a]))
        past = h2h[key][-window:]

        if past:
            # From home team perspective
            h2h_rate = sum(1 for r in past if r == h) / len(past)
        else:
            h2h_rate = 0.5  # no history → neutral

        h2h_rates.append(h2h_rate)

        winner = h if row["home_score"] > row["away_score"] else \
                 (a if row["away_score"] > row["home_score"] else "draw")
        h2h[key].append(winner)

    df = df.copy()
    df["h2h_home_winrate"] = h2h_rates
    return df


# ──────────────────────────────────────────────
# STEP 4 – MODEL TRAINING
# ──────────────────────────────────────────────

FEATURE_COLS = [
    "elo_diff",
    "elo_home",
    "elo_away",
    "neutral",
    "is_wc",
    "home_form_scored",
    "home_form_conceded",
    "away_form_scored",
    "away_form_conceded",
    "form_diff",
    "h2h_home_winrate",
]


def train_model(feat_df: pd.DataFrame) -> xgb.XGBClassifier:
    print("🤖  Training XGBoost classifier …")

    X = feat_df[FEATURE_COLS].fillna(0)
    y = feat_df["outcome"]  # 0=away, 1=draw, 2=home

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"    CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    model.fit(X, y)
    y_pred = model.predict(X)
    print("\n    Training set report:")
    print(classification_report(y, y_pred,
                                target_names=["Away win", "Draw", "Home win"],
                                zero_division=0))
    return model


# ──────────────────────────────────────────────
# STEP 5 – PREDICTION ENGINE
# ──────────────────────────────────────────────

def predict_match(
    home: str,
    away: str,
    elo: EloRatingSystem,
    model: xgb.XGBClassifier,
    feat_df: pd.DataFrame,
    neutral: bool = True,
    is_wc: bool = True,
) -> dict:
    """
    Predict probabilities and expected goals for a single match.
    Returns a dict with full prediction details.
    """

    def team_form(team: str, window: int = 5) -> tuple[float, float]:
        team_matches = feat_df[
            (feat_df["home_team"] == team) | (feat_df["away_team"] == team)
        ].tail(window * 2)
        scored, conceded = [], []
        for _, r in team_matches.iterrows():
            if r["home_team"] == team:
                scored.append(r["home_score"]); conceded.append(r["away_score"])
            else:
                scored.append(r["away_score"]); conceded.append(r["home_score"])
        return (
            np.mean(scored[-window:])   if scored   else 1.5,
            np.mean(conceded[-window:]) if conceded else 1.2,
        )

    def h2h_rate(home: str, away: str, window: int = 10) -> float:
        mask = (
            ((feat_df["home_team"] == home) & (feat_df["away_team"] == away)) |
            ((feat_df["home_team"] == away) & (feat_df["away_team"] == home))
        )
        past = feat_df[mask].tail(window)
        if past.empty:
            return 0.5
        wins = sum(
            1 for _, r in past.iterrows()
            if (r["home_team"] == home and r["home_score"] > r["away_score"]) or
               (r["away_team"] == home and r["away_score"] > r["home_score"])
        )
        return wins / len(past)

    elo_h = elo.ratings.get(home, EloRatingSystem.INITIAL_ELO)
    elo_a = elo.ratings.get(away, EloRatingSystem.INITIAL_ELO)
    elo_diff = elo_h - elo_a + (0 if neutral else 100)

    h_sc, h_cc = team_form(home)
    a_sc, a_cc = team_form(away)

    X = pd.DataFrame([{
        "elo_diff":             elo_diff,
        "elo_home":             elo_h,
        "elo_away":             elo_a,
        "neutral":              int(neutral),
        "is_wc":                int(is_wc),
        "home_form_scored":     h_sc,
        "home_form_conceded":   h_cc,
        "away_form_scored":     a_sc,
        "away_form_conceded":   a_cc,
        "form_diff":            (h_sc - h_cc) - (a_sc - a_cc),
        "h2h_home_winrate":     h2h_rate(home, away),
    }])

    probs = model.predict_proba(X)[0]  # [away, draw, home]
    p_away, p_draw, p_home = float(probs[0]), float(probs[1]), float(probs[2])

    # Expected goals via Dixon-Coles-style attack/defence balance
    # Using form-based lambda with Elo adjustment
    elo_factor = 10 ** (elo_diff / 800)
    xg_home = max(0.3, h_sc * elo_factor * 0.7 + (1 - a_cc / 2) * 0.3)
    xg_away = max(0.3, a_sc / elo_factor * 0.7 + (1 - h_cc / 2) * 0.3)

    # Most likely outcome → predicted winner + confidence (its probability)
    outcomes = {"home": p_home, "draw": p_draw, "away": p_away}
    best = max(outcomes, key=lambda k: outcomes[k])
    winner = {"home": home, "away": away, "draw": "Draw"}[best]

    return {
        "home":       home,
        "away":       away,
        "elo_home":   round(elo_h, 0),
        "elo_away":   round(elo_a, 0),
        "p_home":     round(p_home, 3),
        "p_draw":     round(p_draw, 3),
        "p_away":     round(p_away, 3),
        "xg_home":    round(float(xg_home), 2),
        "xg_away":    round(float(xg_away), 2),
        "score_home": int(round(xg_home)),
        "score_away": int(round(xg_away)),
        "winner":     winner,
        "confidence": round(outcomes[best], 3),
        "favorite":   home if p_home > p_away else away,
    }


# ──────────────────────────────────────────────
# STEP 6 – VISUALISATION
# ──────────────────────────────────────────────

COLORS = {
    "home":    "#2563EB",   # blue
    "draw":    "#64748B",   # gray
    "away":    "#DC2626",   # red
    "bg":      "#0F172A",   # dark bg (YouTube thumbnail style)
    "text":    "#F8FAFC",
    "accent":  "#F59E0B",   # amber
}


def plot_match_prediction(pred: dict, save_path: Path | None = None):
    """
    Generate a YouTube-thumbnail-ready match prediction card.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             facecolor=COLORS["bg"])
    fig.suptitle(
        f"AI MATCH PREDICTION  |  FIFA World Cup 2026",
        color=COLORS["accent"], fontsize=13, fontweight="bold", y=0.97,
    )

    # ── Left: probability bar chart ──────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(COLORS["bg"])
    labels = [pred["home"], "Draw", pred["away"]]
    probs  = [pred["p_home"], pred["p_draw"], pred["p_away"]]
    colors = [COLORS["home"], COLORS["draw"], COLORS["away"]]
    bars   = ax1.barh(labels, [p * 100 for p in probs], color=colors,
                      height=0.5, edgecolor="none")

    for bar, p in zip(bars, probs):
        ax1.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{p * 100:.1f}%", va="center", color=COLORS["text"], fontsize=14,
            fontweight="bold",
        )

    ax1.set_xlim(0, 105)
    ax1.set_xlabel("Win probability (%)", color=COLORS["text"], fontsize=11)
    ax1.tick_params(colors=COLORS["text"], labelsize=12)
    ax1.spines[:].set_visible(False)
    ax1.set_title("Win probabilities", color=COLORS["text"],
                  fontsize=12, pad=10)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.xaxis.set_tick_params(color=COLORS["text"])

    # ── Right: expected goals gauge ───────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(COLORS["bg"])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    # xG circles
    for x, xg, label, color in [
        (0.25, pred["xg_home"], pred["home"], COLORS["home"]),
        (0.75, pred["xg_away"], pred["away"], COLORS["away"]),
    ]:
        circle = plt.Circle((x, 0.55), 0.18, color=color, alpha=0.25)
        ax2.add_patch(circle)
        circle_border = plt.Circle((x, 0.55), 0.18,
                                   color=color, fill=False, lw=2)
        ax2.add_patch(circle_border)
        ax2.text(x, 0.55, f"{xg:.2f}", ha="center", va="center",
                 color=COLORS["text"], fontsize=22, fontweight="bold")
        ax2.text(x, 0.30, label, ha="center", va="center",
                 color=COLORS["text"], fontsize=11, fontweight="bold")

    ax2.text(0.5, 0.92, "Expected Goals (xG)", ha="center",
             color=COLORS["text"], fontsize=12)
    ax2.text(0.5, 0.55, "vs", ha="center", va="center",
             color=COLORS["accent"], fontsize=16, fontweight="bold")

    # Elo ratings
    ax2.text(0.25, 0.12, f"Elo: {pred['elo_home']:.0f}",
             ha="center", color=COLORS["home"], fontsize=10)
    ax2.text(0.75, 0.12, f"Elo: {pred['elo_away']:.0f}",
             ha="center", color=COLORS["away"], fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"    📸 Chart saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_elo_rankings(elo: EloRatingSystem, save_path: Path | None = None,
                      top_n: int = 20):
    """Bar chart of top-N team Elo ratings — great for YouTube intros."""
    top = elo.top_n(top_n)

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    bar_colors = [
        COLORS["accent"] if i < 3 else COLORS["home"]
        for i in range(len(top))
    ]
    bars = ax.barh(top["team"][::-1], top["elo"][::-1],
                   color=bar_colors[::-1], height=0.65, edgecolor="none")

    for bar, elo_val in zip(bars, top["elo"][::-1]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{elo_val:.0f}", va="center", color=COLORS["text"],
                fontsize=9)

    ax.set_xlabel("Elo Rating", color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"])
    ax.spines[:].set_visible(False)
    ax.set_title("AI Power Rankings — FIFA World Cup 2026",
                 color=COLORS["accent"], fontsize=14, fontweight="bold", pad=12)

    gold  = mpatches.Patch(color=COLORS["accent"], label="Top 3")
    other = mpatches.Patch(color=COLORS["home"],   label="Top 20")
    ax.legend(handles=[gold, other], facecolor=COLORS["bg"],
              labelcolor=COLORS["text"], fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"    📸 Rankings chart saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# STEP 7 – TOURNAMENT SIMULATOR
# ──────────────────────────────────────────────

def simulate_tournament(
    elo: EloRatingSystem,
    model: xgb.XGBClassifier,
    feat_df: pd.DataFrame,
    n_sims: int = 10_000,
) -> pd.DataFrame:
    """
    Monte Carlo simulation of the full tournament.
    Returns championship probability for every team.
    """
    print(f"\n🏆  Running {n_sims:,} tournament simulations …")

    # Simplified: use teams from WC2026_MATCHES
    all_teams = sorted(set(
        t for m in WC2026_MATCHES for t in [m[0], m[1]]
    ))

    wins = defaultdict(int)

    for _ in range(n_sims):
        # Sample winner of each group match and advance top 2 per group
        # (simplified bracket: randomly pick 8 QF matchups from group winners)
        remaining = list(all_teams)

        # Knock out progressively until 1 team remains
        while len(remaining) > 1:
            next_round = []
            rng = np.random.default_rng()
            rng.shuffle(remaining)
            for i in range(0, len(remaining) - 1, 2):
                h, a = remaining[i], remaining[i + 1]
                pred = predict_match(h, a, elo, model, feat_df,
                                     neutral=True, is_wc=True)
                roll = rng.random()
                if roll < pred["p_home"]:
                    next_round.append(h)
                elif roll < pred["p_home"] + pred["p_draw"]:
                    # Penalty shootout: 50/50
                    next_round.append(h if rng.random() > 0.5 else a)
                else:
                    next_round.append(a)
            if len(remaining) % 2 == 1:
                next_round.append(remaining[-1])  # bye
            remaining = next_round

        wins[remaining[0]] += 1

    results = pd.DataFrame([
        {"team": t, "championship_prob": round(wins[t] / n_sims * 100, 2)}
        for t in sorted(all_teams, key=lambda x: -wins[x])
    ])
    return results


def plot_championship_probs(sim_results: pd.DataFrame,
                            save_path: Path | None = None):
    top = sim_results.head(12)
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    bar_colors = [
        COLORS["accent"] if i == 0 else
        ("#94A3B8" if i < 3 else COLORS["home"])
        for i in range(len(top))
    ]
    ax.bar(top["team"], top["championship_prob"],
           color=bar_colors, edgecolor="none")

    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(i, row["championship_prob"] + 0.2,
                f"{row['championship_prob']:.1f}%",
                ha="center", color=COLORS["text"], fontsize=9,
                fontweight="bold")

    ax.set_ylabel("Championship probability (%)", color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"], axis="both")
    ax.spines[:].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    ax.set_title("Who Will Win World Cup 2026? — AI Simulation",
                 color=COLORS["accent"], fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"    📸 Simulation chart saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def print_prediction(pred: dict):
    bar = "█"
    total = 40
    h_bar = bar * round(pred["p_home"] * total)
    d_bar = bar * round(pred["p_draw"] * total)
    a_bar = bar * round(pred["p_away"] * total)

    print(f"\n  ⚽  {pred['home']:>16}  vs  {pred['away']:<16}")
    print(f"  {'─'*52}")
    print(f"  {'Home win':<12} {h_bar:<40} {pred['p_home']*100:5.1f}%  (Elo {pred['elo_home']:.0f})")
    print(f"  {'Draw':<12} {d_bar:<40} {pred['p_draw']*100:5.1f}%")
    print(f"  {'Away win':<12} {a_bar:<40} {pred['p_away']*100:5.1f}%  (Elo {pred['elo_away']:.0f})")
    print(f"  {'─'*52}")
    print(f"  xG: {pred['home']} {pred['xg_home']:.2f} – {pred['xg_away']:.2f} {pred['away']}")
    print(f"  🔢  Predicted score: {pred['home']} {pred['score_home']}–{pred['score_away']} {pred['away']}")
    if pred["winner"] == "Draw":
        print(f"  🏅  Prediction: Draw  ({pred['confidence']*100:.1f}% confidence)")
    else:
        print(f"  🏅  Predicted winner: {pred['winner']}  ({pred['confidence']*100:.1f}% confidence)")


def _slug(team: str) -> str:
    """Filesystem-safe slug for chart filenames (e.g. 'Saudi Arabia' → 'saudi_arabia')."""
    return re.sub(r"[^a-z0-9]+", "_", team.lower()).strip("_")


def print_fixtures(fixtures: pd.DataFrame):
    """List upcoming WC2026 fixtures with index numbers for --game selection."""
    if fixtures.empty:
        print("\n📅  No upcoming fixtures with confirmed teams were found.")
        return
    print("\n📅  Upcoming WC2026 fixtures (use --game N to predict one):")
    for i, r in fixtures.iterrows():
        stage = r.get("group") or r.get("round") or ""
        print(f"  [{i:>2}]  {r['date'].date()}  "
              f"{r['home_team']:>22} vs {r['away_team']:<22}  {stage}")


def main():
    parser = argparse.ArgumentParser(description="FIFA 2026 Match Predictor")
    parser.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"),
                        help="Predict a specific match")
    parser.add_argument("--upcoming", action="store_true",
                        help="List upcoming WC2026 fixtures (with index numbers)")
    parser.add_argument("--game", type=int, metavar="N",
                        help="Predict upcoming fixture number N (see --upcoming)")
    parser.add_argument("--simulate", action="store_true",
                        help="Run full tournament simulation")
    parser.add_argument("--rankings", action="store_true",
                        help="Show Elo power rankings")
    args = parser.parse_args()

    # ── Pipeline ──────────────────────────────
    df      = load_results()
    elo     = EloRatingSystem().fit(df)
    feat_df = build_features(df, elo)
    model   = train_model(feat_df)

    # ── Rankings ──────────────────────────────
    print("\n📊  Current AI Power Rankings (Elo):")
    print(elo.top_n(15).to_string(index=False))
    plot_elo_rankings(elo, save_path=VIS_DIR / "elo_rankings.png")

    # ── Upcoming fixtures (live from openfootball) ───────────
    fixtures = upcoming_fixtures()
    if args.upcoming:
        print_fixtures(fixtures)

    # ── Match predictions ─────────────────────
    if args.game is not None:
        if fixtures.empty or not (0 <= args.game < len(fixtures)):
            print_fixtures(fixtures)
            print(f"\n❌  No upcoming fixture #{args.game}. Pick an index above.")
            return
        row = fixtures.iloc[args.game]
        home, away = row["home_team"], row["away_team"]
        print(f"\n🔮  Predicting upcoming fixture #{args.game}: "
              f"{home} vs {away} on {row['date'].date()}")
        pred = predict_match(home, away, elo, model, feat_df,
                             neutral=bool(row["neutral"]), is_wc=True)
        print_prediction(pred)
        plot_match_prediction(pred,
            save_path=VIS_DIR / f"pred_{_slug(home)}_{_slug(away)}.png")
    elif args.match:
        home, away = normalize_team(args.match[0]), normalize_team(args.match[1])
        pred = predict_match(home, away, elo, model, feat_df)
        print_prediction(pred)
        plot_match_prediction(pred,
            save_path=VIS_DIR / f"pred_{_slug(home)}_{_slug(away)}.png")
    elif args.upcoming:
        pass  # already listed fixtures above; nothing more to predict
    else:
        # Demo: predict a selection of high-interest WC26 group matches
        demo_matches = [
            ("France",    "Argentina"),
            ("Brazil",    "Germany"),
            ("England",   "Spain"),
            ("USA",       "Canada"),
        ]
        print("\n🔮  Pre-tournament predictions for key group-stage clashes:\n")
        for home, away in demo_matches:
            pred = predict_match(home, away, elo, model, feat_df)
            print_prediction(pred)
            plot_match_prediction(
                pred,
                save_path=VIS_DIR / f"pred_{_slug(home)}_{_slug(away)}.png",
            )

    # ── Tournament simulation ──────────────────
    if args.simulate:
        sim = simulate_tournament(elo, model, feat_df, n_sims=10_000)
        print("\n🏆  Championship probabilities:")
        print(sim.head(12).to_string(index=False))
        plot_championship_probs(sim,
            save_path=VIS_DIR / "championship_probs.png")


if __name__ == "__main__":
    main()
