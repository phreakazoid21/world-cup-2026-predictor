# ⚽ FIFA World Cup 2026 — AI Match Prediction Model
AI match prediction model for FIFA World Cup 2026. Elo ratings + XGBoost trained on 50,000+ international matches.
> A complete Python pipeline for predicting match outcomes, generating
> expected goals, and running Monte Carlo tournament simulations.
> Built for YouTube creators who want AI-powered soccer analysis.

---

## 🚀 Quick Start

```bash
# 1. Clone / download this folder
cd fifa_predictor

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Run the demo (downloads data automatically on first run)
python predictor.py

# 4. Predict a specific match
python predictor.py --match "Brazil" "France"

# 5. List upcoming WC2026 fixtures (live, from openfootball)
python predictor.py --upcoming

# 6. Predict an upcoming fixture by its index number
python predictor.py --game 0

# 7. Full tournament simulation (10,000 runs)
python predictor.py --simulate
```

Each prediction reports the **predicted winner + win probability**, a
**predicted scoreline**, and a **confidence** number (the probability of the
most likely outcome).

Charts are saved to the `visuals/` folder — ready to use in videos.

---

## 📁 Project Structure

```
fifa_predictor/
├── predictor.py            ← Main model (all-in-one)
├── tutorial_notebook.ipynb ← Jupyter walkthrough for YouTube tutorials
├── requirements.txt
├── data/
│   └── results.csv         ← Auto-downloaded on first run
├── models/                 ← Saved model files
└── visuals/                ← Generated charts (PNG, 150 dpi)
    ├── elo_rankings.png
    ├── championship_probs.png
    ├── feature_importance.png
    └── pred_brazil_france.png  (etc.)
```

---

## 🧠 How It Works

### 1. Data
- **Historical matches:** 50,000+ international results from 1990–2026
  - Source: [martj42/international_results](https://github.com/martj42/international_results) (free, no key)
- **Live WC26 results + schedule:** pulled at runtime from
  [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json)
  (public domain, no key). `load_results()` reshapes the feed to the historical
  schema and merges already-played WC2026 matches into the training set, so Elo
  ratings update as the tournament progresses. `upcoming_fixtures()` exposes the
  not-yet-played games you can predict with `--upcoming` / `--game`.
- **During tournament:** plug in [BALLDONTLIE FIFA API](https://fifa.balldontlie.io/) (free tier)

### 2. Elo Rating System
Dynamic Elo ratings updated after every match since 1990.  
Key design choices:
- Tournament importance weights (World Cup k=60, Friendly k=20)
- Home advantage: +100 Elo unless neutral venue
- Goal-difference multiplier (winning 3-0 counts more than 1-0)

### 3. Features Used by the Model

| Feature | Description |
|---|---|
| `elo_diff` | Elo gap between teams (adjusted for venue) |
| `elo_home / elo_away` | Absolute ratings |
| `home_form_scored` | Avg goals scored, last 5 matches |
| `home_form_conceded` | Avg goals conceded, last 5 matches |
| `form_diff` | Net form advantage |
| `h2h_home_winrate` | Head-to-head record, last 10 meetings |
| `is_wc` | Tournament context (World Cup = 1) |
| `neutral` | Neutral venue flag |

### 4. Model
- **Algorithm:** XGBoost multi-class classifier
- **Target:** 3 outcomes — Home Win / Draw / Away Win
- **Validation:** 5-fold stratified cross-validation
- **Expected accuracy:** ~55–58% (benchmark: ~45% for random, ~50% for always-pick-favourite)

### 5. Expected Goals (xG)
Form-based xG estimate adjusted by Elo strength ratio using a
simplified Dixon-Coles-style formula. Good enough for video graphics;
upgrade with StatsBomb open data for shot-level xG.

---

## 🎬 YouTube Video Ideas

| Video | Chart to use | When to publish |
|---|---|---|
| "AI Power Rankings – WC2026" | `elo_rankings.png` | Now (pre-tournament) |
| "Who Will Win the World Cup?" | `championship_probs.png` | 2 weeks before kick-off |
| "Brazil vs Germany – AI Prediction" | `pred_brazil_germany.png` | Day before match |
| "Build a Soccer AI in Python" | `feature_importance.png` | Any time (tutorial) |
| "Was AI Right?" recap | Pred vs actual comparison | Day after match |

---

## 🔌 Adding Live Data (During Tournament)

```python
# Option A: BALLDONTLIE FIFA API (free tier, 2018/2022/2026 data)
import requests
r = requests.get(
    "https://api.balldontlie.io/fifa/v1/matches",
    headers={"Authorization": "YOUR_FREE_API_KEY"},
    params={"season": 2026}
)
matches = r.json()["data"]

# Option B: openfootball JSON (no key required)
r = requests.get(
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
)
schedule = r.json()
```

---

## 📈 Improvements for Advanced Videos

- **Shot-level xG:** Pull StatsBomb open data and train a logistic model on shot position + body part
- **Player ratings:** Merge FIFA/SOFIFA ratings as extra features
- **NLP sentiment:** Add Twitter/Reddit fan sentiment score as a feature
- **Live updating:** Re-run Elo after every match and republish predictions
- **Calibration:** Apply Platt scaling to make probabilities better-calibrated

---

## 📄 License
MIT. Use freely for your YouTube channel, course, or SaaS.
Credit appreciated but not required.
