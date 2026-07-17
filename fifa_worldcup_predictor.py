"""
FIFA World Cup 2026 — Winner Prediction Dashboard  (v3 — dropdown fix)
=======================================================================
Run with:  streamlit run fifa_worldcup_predictor.py

REQUIRED DATA FILES (upload in sidebar)
----------------------------------------
FILE 1 ▸ players_fifa25.csv
  Source : Kaggle — "FIFA 25 Complete Player Dataset"
  URL    : https://www.kaggle.com/datasets/stefanoleone992/fifa-25-complete-player-dataset
  Key cols: nationality_name, player_positions, overall, pace, shooting,
            passing, dribbling, defending, physic, value_eur, age

FILE 2 ▸ team_data.csv
  Build it yourself (one row per qualified team).
  Required col : team
  Optional cols: fifa_ranking, fifa_points, recent_form_pts, squad_avg_age,
                 squad_market_value, host_status (1/0), wc_appearances,
                 confederation, wc_2026_group

FILE 3 ▸ results.csv  ← most important for accuracy
  Source : https://raw.githubusercontent.com/martj42/international-football-results/master/results.csv
  Key cols: date, home_team, away_team, home_score, away_score, tournament

FILE 4 ▸ shootouts.csv  (optional — improves penalty prediction)
  Source : https://raw.githubusercontent.com/martj42/international-football-results/master/shootouts.csv
  Key cols: date, home_team, away_team, winner

FILE 5 ▸ former_names.csv  (optional — maps old country names to current)
  Source : https://raw.githubusercontent.com/martj42/international-football-results/master/former_names.csv
  Key cols: former, current

FIXES IN v3
-----------
- Dropdown reset bug fixed: all interactive widgets now use st.session_state
  keys so selections survive Streamlit reruns triggered by other widgets.
- File uploader labels now show exact expected filename.
- Sidebar shows a clear data-file guide with download links.
- Multiselect in Tournament tab also keyed to session_state.

FIXES IN v4
-----------
- WC_2026_GROUPS replaced with the actual confirmed 48-team, 12-group draw
  (groups A-L), fixing missing teams (Algeria, DR Congo, Cape Verde,
  Curaçao, Jordan, Uzbekistan, Iraq, Ivory Coast, Ghana, etc.) and removing
  the duplicate/incorrect entries (e.g. Costa Rica listed twice, teams that
  didn't actually qualify like Bolivia, Slovenia, Slovakia, Venezuela...).
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import math
import random
import io

# ── third-party ───────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import poisson

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLE
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    layout="wide",
    page_icon="🏆",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --pitch-dark:  #0d1f12;
    --pitch-mid:   #132b1a;
    --pitch-panel: #1a3822;
    --accent-gold: #f5c518;
    --accent-green:#3ddc84;
    --text-main:   #e8f5e9;
    --text-muted:  #8fac92;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--pitch-dark);
    color: var(--text-main);
}
[data-testid="stSidebar"] { background-color: var(--pitch-mid); }
h1 { color: var(--accent-gold) !important; letter-spacing: 0.04em; }
h2, h3 { color: var(--accent-green) !important; }
[data-testid="stMetric"] {
    background: var(--pitch-panel);
    border: 1px solid #2e5c35;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.78rem; }
[data-testid="stMetricValue"] { color: var(--accent-gold) !important; font-size: 1.6rem; }
.stTabs [data-baseweb="tab-list"] { background: var(--pitch-mid); border-radius: 8px; }
.stTabs [data-baseweb="tab"]      { color: var(--text-muted); font-weight: 600; }
.stTabs [aria-selected="true"]    { color: var(--accent-gold) !important; border-bottom: 2px solid var(--accent-gold); }
.stButton > button {
    background: var(--accent-gold); color: #000;
    font-weight: 700; border-radius: 6px; border: none;
}
.stButton > button:hover { background: #ffd740; }
[data-testid="stDataFrame"] { border: 1px solid #2e5c35; border-radius: 8px; }
hr { border-color: #2e5c35; }
.stCaption, small { color: var(--text-muted) !important; }
.stAlert { border-radius: 8px; }

/* Fix selectbox / multiselect background so it's readable on dark theme */
[data-baseweb="select"] > div,
[data-baseweb="popover"] {
    background-color: #1a3822 !important;
    color: #e8f5e9 !important;
}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

# Verified against the official FIFA World Cup 2026 final draw (Dec 5, 2025)
# and the confirmed 48-team qualified list. 12 groups (A-L) x 4 teams = 48.
WC_2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

ALL_WC_2026_TEAMS = [t for grp in WC_2026_GROUPS.values() for t in grp]

TOURNAMENT_K = {
    "FIFA World Cup": 50,
    "FIFA World Cup qualification": 30,
    "UEFA Euro": 40,
    "Copa América": 40,
    "African Cup of Nations": 35,
    "CONCACAF Gold Cup": 30,
    "AFC Asian Cup": 30,
    "Friendly": 15,
}
DEFAULT_K    = 25
BASE_LAMBDA  = 1.10
EXP_SCALE    = 0.60
HOME_BONUS   = 0.20
DC_RHO       = -0.13
PEAK_AGE     = 27.5
AGE_HALFWIDTH = 3.5

PLOTLY_TEMPLATE = "plotly_dark"
PAPER_BG = "#0d1f12"
PLOT_BG  = "#132b1a"

# ═════════════════════════════════════════════════════════════════════════════
# SESSION-STATE INITIALISATION
# Must happen before any widget is rendered so keys exist on first load.
# ═════════════════════════════════════════════════════════════════════════════
_SS_DEFAULTS = {
    # Match Predictor tab
    "match_team_a":     "France",
    "match_team_b":     "Croatia",
    "match_n_sims":     5000,
    "match_knockout":   False,
    # Tournament tab
    "tourney_selected": ALL_WC_2026_TEAMS,
    # Results storage
    "match_result":     None,
    "tourney_result":   None,
    "backtest_result":  None,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def read_csv_cached(file_id, file_bytes: bytes) -> pd.DataFrame:
    """Cache keyed on (file name, bytes) so re-uploads bust the cache."""
    return clean_cols(pd.read_csv(io.BytesIO(file_bytes)))


def read_csv(uploaded_file) -> pd.DataFrame:
    return read_csv_cached(uploaded_file.name, uploaded_file.getvalue())


def norm(series: pd.Series, invert: bool = False) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=s.index)
    out = (s - lo) / (hi - lo)
    return 1 - out if invert else out


def age_score_series(series: pd.Series) -> pd.Series:
    return (1 - (series - PEAK_AGE).abs() / AGE_HALFWIDTH).clip(0, 1)


def resolve_name(name: str, name_map: dict) -> str:
    name = str(name).strip()
    return name_map.get(name, name)


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color="#e8f5e9"), title_font=dict(color="#f5c518"),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# DEMO DATA
# ═════════════════════════════════════════════════════════════════════════════

# Approximate real-world strength tiers (roughly June 2026 FIFA ranking order)
# used ONLY to seed believable synthetic demo data. A small amount of random
# noise is layered on top of each team's tier so the demo isn't perfectly
# deterministic, but a minnow like Jordan or Curaçao can no longer randomly
# roll a quality score on par with Argentina or France.
DEMO_TEAM_TIER = {
    # tier ~1.6-2.0: traditional top contenders
    "France": 2.0, "Spain": 1.95, "Argentina": 1.9, "England": 1.85,
    "Portugal": 1.8, "Brazil": 1.75, "Netherlands": 1.55, "Belgium": 1.5,
    "Germany": 1.5, "Croatia": 1.35,
    # tier ~0.6-1.2: strong, regular knockout-stage teams
    "Italy": 1.2, "Morocco": 1.1, "Switzerland": 1.05, "Colombia": 1.0,
    "Uruguay": 1.0, "United States": 0.85, "Senegal": 0.8, "Japan": 0.8,
    "Mexico": 0.75, "Ecuador": 0.7, "Austria": 0.7, "Iran": 0.65,
    "Egypt": 0.6, "Canada": 0.6, "Ivory Coast": 0.6, "Australia": 0.55,
    "Norway": 0.55, "Algeria": 0.55, "Paraguay": 0.5, "Tunisia": 0.5,
    "Scotland": 0.5, "Turkey": 0.5,
    # tier ~0.0-0.4: mid-table / first-time-in-a-while qualifiers
    "South Korea": 0.4, "Saudi Arabia": 0.35, "Ghana": 0.3, "Panama": 0.3,
    "Qatar": 0.3, "Sweden": 0.3, "Czech Republic": 0.25, "Bosnia and Herzegovina": 0.2,
    "South Africa": 0.15, "Iraq": 0.1, "Uzbekistan": 0.05, "DR Congo": 0.05,
    # tier ~ -0.6 to -0.2: lower-ranked debutants / smaller federations
    "Jordan": -0.2, "New Zealand": -0.2, "Cape Verde": -0.25, "Curacao": -0.35,
    "Haiti": -0.35, "Venezuela": -0.3,
}
DEMO_DEFAULT_TIER = 0.0   # any team not listed above gets an average tier


def make_demo_player_data(teams: list, n_per_team: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    # quality = real-world strength tier + small noise, NOT pure noise.
    # This keeps the synthetic demo data roughly realistic (Argentina ranks
    # far above Jordan, etc.) instead of randomly equalising every team.
    team_quality = {
        t: DEMO_TEAM_TIER.get(t, DEMO_DEFAULT_TIER) + rng.normal(0, 0.25)
        for t in teams
    }
    rows = []
    for t in teams:
        q = team_quality[t]
        # Force a minimum number of goalkeepers per squad so gk_quality is
        # never undefined for a team (a real squad always has GKs; pure
        # random position sampling could occasionally draw zero in 23 picks,
        # which produced a NaN that silently poisoned defense_strength for
        # every team in the strength-index weighted sum).
        n_gk = 3
        positions = ["GK"] * n_gk + list(
            rng.choice(["DEF", "MID", "FWD"], size=n_per_team - n_gk, p=[0.40, 0.34, 0.26])
        )
        for pos in positions:
            rows.append({
                "nationality_name": t,
                "player_positions": pos,
                "age":      int(rng.integers(19, 34)),
                "overall":  int(np.clip(rng.normal(70 + 9 * q, 4), 50, 95)),
                "value_eur": max(0, rng.exponential(6e6) * (1 + 1.2 * q)),
                "pace":      int(np.clip(rng.normal(68 + 5 * q, 7), 35, 99)),
                "shooting":  int(np.clip(rng.normal(63 + 7 * q, 8), 30, 99)) if pos != "GK" else 20,
                "passing":   int(np.clip(rng.normal(68 + 7 * q, 6), 35, 99)),
                "dribbling": int(np.clip(rng.normal(66 + 7 * q, 7), 30, 99)),
                "defending": int(np.clip(rng.normal(64 + 6 * q, 7), 25, 99)),
                "physic":    int(np.clip(rng.normal(67 + 5 * q, 6), 35, 99)),
                "saves_per90": max(0, rng.normal(3.0 + 0.7 * q, 0.6)) if pos == "GK" else 0,
            })
    return pd.DataFrame(rows)


def make_demo_team_data(teams: list) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    quality = {
        t: DEMO_TEAM_TIER.get(t, DEMO_DEFAULT_TIER) + rng.normal(0, 0.2)
        for t in teams
    }
    # Rank teams by their (noisy) quality score so fifa_ranking is actually
    # correlated with strength instead of being a pure random shuffle.
    ranking_order = sorted(teams, key=lambda t: quality[t], reverse=True)
    rank_lookup = {t: i + 1 for i, t in enumerate(ranking_order)}

    return pd.DataFrame({
        "team":               teams,
        "fifa_ranking":       [rank_lookup[t] for t in teams],
        "fifa_points":        [max(800, 1300 + 220 * quality[t] + rng.normal(0, 25)) for t in teams],
        "recent_form_pts":    [max(0, 50 + 16 * quality[t] + rng.normal(0, 6)) for t in teams],
        "squad_avg_age":      [float(np.clip(rng.normal(27.2, 1.5), 22, 33)) for _ in teams],
        "squad_market_value": [float(np.clip(40 + 260 * max(quality[t], -0.5) + rng.exponential(40), 15, 1800)) for t in teams],
        "host_status":        [1 if t in {"United States", "Canada", "Mexico"} else 0 for t in teams],
        "wc_appearances":     [int(np.clip(8 + 8 * quality[t] + rng.integers(-2, 3), 0, 22)) for t in teams],
    })


# ═════════════════════════════════════════════════════════════════════════════
# ELO ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def k_factor(tournament_name, margin: int) -> float:
    name = str(tournament_name) if isinstance(tournament_name, str) else ""
    base = DEFAULT_K
    for key, val in TOURNAMENT_K.items():
        if key.lower() in name.lower():
            base = val
            break
    mov = math.log(abs(margin) + 1.5)
    return min(base * mov, 80)


@st.cache_data(show_spinner=False)
def compute_elo(results_json: str, years_back: int, start_rating: float = 1500.0) -> pd.DataFrame:
    """Accepts JSON string so Streamlit can hash it cleanly."""
    df = pd.read_json(io.StringIO(results_json))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"]).sort_values("date")
    if years_back > 0:
        cutoff = df["date"].max() - pd.Timedelta(days=365 * years_back)
        df = df[df["date"] >= cutoff]

    ratings: dict[str, float] = {}
    match_count: dict[str, int] = {}

    def get(t):
        return ratings.get(t, start_rating)

    for row in df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        hs, as_ = float(row.home_score), float(row.away_score)
        rh, ra = get(h), get(a)
        exp_h  = 1.0 / (1.0 + 10.0 ** ((ra - rh) / 400.0))
        actual = 1.0 if hs > as_ else (0.0 if as_ > hs else 0.5)
        k      = k_factor(getattr(row, "tournament", ""), int(abs(hs - as_)))
        delta  = k * (actual - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
        match_count[h] = match_count.get(h, 0) + 1
        match_count[a] = match_count.get(a, 0) + 1

    return pd.DataFrame({
        "team":        list(ratings),
        "elo_rating":  list(ratings.values()),
        "elo_matches": [match_count[t] for t in ratings],
    }).sort_values("elo_rating", ascending=False).reset_index(drop=True)


def get_shootout_prob(team_a: str, team_b: str, shootouts_df) -> float | None:
    if shootouts_df is None:
        return None
    h2h = shootouts_df[
        ((shootouts_df.home_team == team_a) & (shootouts_df.away_team == team_b)) |
        ((shootouts_df.home_team == team_b) & (shootouts_df.away_team == team_a))
    ]
    if len(h2h) >= 2:
        return float((h2h["winner"] == team_a).mean())
    def rate(team):
        sub = shootouts_df[(shootouts_df.home_team == team) | (shootouts_df.away_team == team)]
        return float((sub["winner"] == team).mean()) if len(sub) >= 2 else None
    ra, rb = rate(team_a), rate(team_b)
    if ra is None or rb is None:
        return None
    return ra / (ra + rb) if (ra + rb) > 0 else 0.5


# ═════════════════════════════════════════════════════════════════════════════
# PLAYER → SQUAD QUALITY
# ═════════════════════════════════════════════════════════════════════════════

PLAYER_COL_MAP = {
    "team":     ["nationality_name", "team", "nation", "country"],
    "pos":      ["player_positions", "position", "pos"],
    "overall":  ["overall", "rating"],
    "value":    ["value_eur", "value", "market_value"],
    "shooting": ["shooting"],
    "passing":  ["passing", "pass_accuracy"],
    "defending":["defending"],
    "pace":     ["pace"],
    "dribbling":["dribbling"],
    "physic":   ["physic"],
    "saves":    ["saves_per90", "saves"],
}


def find_col(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def aggregate_squad_quality(player_df: pd.DataFrame) -> pd.DataFrame:
    tc = find_col(player_df, PLAYER_COL_MAP["team"])
    if tc is None:
        st.error("Player file must have one of: " + ", ".join(PLAYER_COL_MAP["team"]))
        st.stop()
    pc  = find_col(player_df, PLAYER_COL_MAP["pos"])
    oc  = find_col(player_df, PLAYER_COL_MAP["overall"])
    vc  = find_col(player_df, PLAYER_COL_MAP["value"])
    shc = find_col(player_df, PLAYER_COL_MAP["shooting"])
    pac = find_col(player_df, PLAYER_COL_MAP["passing"])
    dc  = find_col(player_df, PLAYER_COL_MAP["defending"])
    pcc = find_col(player_df, PLAYER_COL_MAP["pace"])
    drc = find_col(player_df, PLAYER_COL_MAP["dribbling"])
    phc = find_col(player_df, PLAYER_COL_MAP["physic"])
    sc  = find_col(player_df, PLAYER_COL_MAP["saves"])

    rows = []
    for team, g in player_df.groupby(tc):
        row = {"team": team, "squad_size": len(g)}
        if oc:
            row["avg_overall"]  = g[oc].mean()
            row["top11_overall"]= g[oc].nlargest(11).mean()
        if vc:
            row["total_value_eur"] = g[vc].sum()
        if shc: row["avg_shooting"]  = g[shc].mean()
        if pac: row["avg_passing"]   = g[pac].mean()
        if dc:  row["avg_defending"] = g[dc].mean()
        if pcc: row["avg_pace"]      = g[pcc].mean()
        if drc: row["avg_dribbling"] = g[drc].mean()
        if phc: row["avg_physic"]    = g[phc].mean()
        if sc and pc:
            gks = g[g[pc].str.contains("GK", na=False)]
            if len(gks):
                row["gk_quality"] = gks[sc].mean()
        if "age" in g.columns:
            row["squad_avg_age_player"] = g["age"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING & STRENGTH INDEX
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_SPECS = {
    # key: (invert, default_weight, group, label)
    "elo_rating":           (False, 40, "overall",  "Elo Rating (match history)"),
    "fifa_ranking":         (True,   8, "overall",  "FIFA Ranking (lower = better)"),
    "fifa_points":          (False, 10, "overall",  "FIFA Points"),
    "recent_form_pts":      (False, 10, "overall",  "Recent Form Points"),
    "squad_market_value":   (False,  6, "overall",  "Team Market Value (M€)"),
    "total_value_eur":      (False,  6, "overall",  "Squad Total Value (€)"),
    "wc_appearances":       (False,  5, "overall",  "World Cup Appearances"),
    "host_status":          (False,  4, "overall",  "Host Nation Bonus"),
    "squad_avg_age":        (None,   4, "overall",  "Squad Avg Age (team file)"),
    "squad_avg_age_player": (None,   4, "overall",  "Squad Avg Age (player file)"),
    "avg_overall":          (False, 12, "overall",  "Avg Player Overall Rating"),
    "top11_overall":        (False, 14, "overall",  "Top-11 Avg Overall Rating"),
    "avg_shooting":         (False, 10, "attack",   "Avg Shooting Rating"),
    "avg_dribbling":        (False,  6, "attack",   "Avg Dribbling Rating"),
    "avg_pace":             (False,  4, "attack",   "Avg Pace Rating"),
    "avg_defending":        (False, 10, "defense",  "Avg Defending Rating"),
    "avg_physic":           (False,  5, "defense",  "Avg Physic Rating"),
    "gk_quality":           (False,  5, "defense",  "GK Quality (saves per 90)"),
    "avg_passing":          (False,  6, "overall",  "Avg Passing Rating"),
}


def build_strength_index(teams_df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = teams_df.copy()
    available = [f for f in FEATURE_SPECS if f in df.columns and df[f].notna().sum() > 1]

    norms = {}
    atk_parts, def_parts = [], []

    for f in available:
        invert, _, group, _ = FEATURE_SPECS[f]
        if f in ("squad_avg_age", "squad_avg_age_player"):
            comp = age_score_series(df[f])
        elif invert is None:
            comp = norm(df[f])
        else:
            comp = norm(df[f], invert=invert)
        # A team missing this one feature (e.g. a squad with no GK data,
        # or an upload with a sparse column) must not poison every other
        # team's score: fill any leftover NaN with the league-average
        # component value (0.5 on the normalised 0-1 scale) rather than
        # letting it propagate through the weighted sum below.
        comp = comp.fillna(0.5)
        norms[f] = comp
        w = weights.get(f, 0)
        if group == "attack":
            atk_parts.append((comp, w))
        elif group == "defense":
            def_parts.append((comp, w))

    total_w = sum(weights.get(f, 0) for f in available) or 1
    overall = sum(norms[f] * weights.get(f, 0) for f in available) / total_w * 100

    def sub(parts):
        sw = sum(w for _, w in parts)
        return sum(c * w for c, w in parts) / sw if sw else overall / 100

    df["strength_index"]   = overall.round(2)
    df["attack_strength"]  = (sub(atk_parts) * 100).round(2) if atk_parts else (overall * 0.5)
    df["defense_strength"] = (sub(def_parts) * 100).round(2) if def_parts else (overall * 0.5)
    return df.sort_values("strength_index", ascending=False).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# MATCH MODEL
# ═════════════════════════════════════════════════════════════════════════════

def expected_goals(atk: float, dfs: float, home: bool = False) -> float:
    diff = (atk - dfs) / 100.0
    lam  = BASE_LAMBDA * math.exp(EXP_SCALE * diff)
    if home:
        lam += HOME_BONUS
    return max(0.10, min(lam, 3.50))


def dc_correction(ga: int, gb: int, la: float, lb: float, rho: float = DC_RHO) -> float:
    if ga == 0 and gb == 0: return 1 - la * lb * rho
    if ga == 0 and gb == 1: return 1 + la * rho
    if ga == 1 and gb == 0: return 1 + lb * rho
    if ga == 1 and gb == 1: return 1 - rho
    return 1.0


def simulate_match(
    team_a: str, team_b: str, teams_df: pd.DataFrame,
    knockout: bool = False, host_team: str | None = None,
    rng=None, shootouts_df=None,
) -> tuple[int, int, str | None]:
    rng = rng if rng is not None else np.random.default_rng()
    ra = teams_df.loc[teams_df.team == team_a].iloc[0]
    rb = teams_df.loc[teams_df.team == team_b].iloc[0]
    la = expected_goals(ra["attack_strength"], rb["defense_strength"], home=(team_a == host_team))
    lb = expected_goals(rb["attack_strength"], ra["defense_strength"], home=(team_b == host_team))

    MG = 9
    joint = np.outer(
        [poisson.pmf(g, la) for g in range(MG)],
        [poisson.pmf(g, lb) for g in range(MG)],
    )
    for i in range(2):
        for j in range(2):
            joint[i, j] *= dc_correction(i, j, la, lb)
    joint /= joint.sum()
    idx = rng.choice(MG * MG, p=joint.ravel())
    ga, gb = divmod(int(idx), MG)

    winner = None
    if ga > gb:
        winner = team_a
    elif gb > ga:
        winner = team_b
    elif knockout:
        p_a = get_shootout_prob(team_a, team_b, shootouts_df)
        if p_a is None:
            sa, sb = float(ra["strength_index"]), float(rb["strength_index"])
            p_a = sa / (sa + sb + 1e-9)
        winner = team_a if rng.random() < p_a else team_b
    return int(ga), int(gb), winner


def match_outcome_probs(
    team_a: str, team_b: str, teams_df: pd.DataFrame,
    n_sims: int = 5000, host_team: str | None = None,
) -> tuple[float, float, float]:
    rng = np.random.default_rng()
    wa = wb = d = 0
    for _ in range(n_sims):
        ga, gb, _ = simulate_match(team_a, team_b, teams_df,
                                    knockout=False, host_team=host_team, rng=rng)
        if ga > gb: wa += 1
        elif gb > ga: wb += 1
        else: d += 1
    return wa / n_sims, d / n_sims, wb / n_sims


# ═════════════════════════════════════════════════════════════════════════════
# TOURNAMENT SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

def simulate_group(group, teams_df, rng, host_team=None, shootouts_df=None):
    pts = {t: 0 for t in group}
    gd  = {t: 0 for t in group}
    gf  = {t: 0 for t in group}
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            ga, gb, winner = simulate_match(a, b, teams_df, knockout=False,
                                             host_team=host_team, rng=rng,
                                             shootouts_df=shootouts_df)
            gf[a] += ga; gf[b] += gb
            gd[a] += ga - gb; gd[b] += gb - ga
            if winner == a: pts[a] += 3
            elif winner == b: pts[b] += 3
            else: pts[a] += 1; pts[b] += 1
    ranking = sorted(group, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
    return ranking, pts


def simulate_knockout_stage(teams, teams_df, rng, host_team=None, shootouts_df=None):
    round_teams = list(teams)
    while len(round_teams) > 1:
        next_round = []
        if len(round_teams) % 2 == 1:
            # bye to highest-ranked team
            bye = teams_df[teams_df.team.isin(round_teams)].iloc[0]["team"]
            round_teams.remove(bye)
            next_round.append(bye)
        for i in range(0, len(round_teams), 2):
            _, _, winner = simulate_match(round_teams[i], round_teams[i + 1], teams_df,
                                           knockout=True, host_team=host_team,
                                           rng=rng, shootouts_df=shootouts_df)
            next_round.append(winner)
        arr = np.array(next_round)
        rng.shuffle(arr)
        round_teams = arr.tolist()
    return round_teams[0] if round_teams else None


def simulate_full_tournament(groups, teams_df, advance_per_group=2, rng=None,
                              host_team=None, shootouts_df=None, include_best_thirds=True):
    rng = rng or np.random.default_rng()
    qualified = []
    thirds = []
    for grp_teams in groups.values():
        present = [t for t in grp_teams if t in teams_df["team"].values]
        if len(present) < 2:
            qualified.extend(present)
            continue
        ranking, pts = simulate_group(present, teams_df, rng, host_team=host_team, shootouts_df=shootouts_df)
        qualified.extend(ranking[:advance_per_group])
        if include_best_thirds and len(ranking) > advance_per_group:
            thirds.append((ranking[advance_per_group], pts.get(ranking[advance_per_group], 0)))
    if include_best_thirds and thirds:
        thirds.sort(key=lambda x: x[1], reverse=True)
        qualified.extend(t for t, _ in thirds[:8])
    if len(qualified) < 2:
        return qualified[0] if qualified else None
    arr = np.array(qualified)
    rng.shuffle(arr)
    return simulate_knockout_stage(arr.tolist(), teams_df, rng, host_team=host_team, shootouts_df=shootouts_df)


@st.cache_data(show_spinner=False)
def run_monte_carlo(
    groups_json: str,
    teams_json: str,
    n_sims: int,
    advance_per_group: int,
    host_team: str | None,
    include_best_thirds: bool,
) -> pd.DataFrame:
    groups   = {k: list(v) for k, v in pd.read_json(io.StringIO(groups_json), typ="series").items()}
    teams_df = pd.read_json(io.StringIO(teams_json))
    rng      = np.random.default_rng(2026)
    all_teams = [t for g in groups.values() for t in g]
    counts = {t: 0 for t in all_teams}
    for _ in range(n_sims):
        champ = simulate_full_tournament(groups, teams_df, advance_per_group=advance_per_group,
                                          rng=rng, host_team=host_team,
                                          include_best_thirds=include_best_thirds)
        if champ:
            counts[champ] = counts.get(champ, 0) + 1
    probs = pd.DataFrame({
        "team":              list(counts),
        "simulations_won":   list(counts.values()),
        "win_probability_%": [round(v / n_sims * 100, 2) for v in counts.values()],
    })
    return probs.sort_values("win_probability_%", ascending=False).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📁 Upload Data Files")
    st.markdown("""
| # | Upload file named | Source |
|---|---|---|
| 1 | **players_fifa25.csv** | [Kaggle FIFA 25](https://www.kaggle.com/datasets/stefanoleone992/fifa-25-complete-player-dataset) |
| 2 | **team_data.csv** | Build yourself (see docs) |
| 3 | **results.csv** | [GitHub (free)](https://raw.githubusercontent.com/martj42/international-football-results/master/results.csv) |
| 4 | **shootouts.csv** | [GitHub (free)](https://raw.githubusercontent.com/martj42/international-football-results/master/shootouts.csv) |
| 5 | **former_names.csv** | [GitHub (optional)](https://raw.githubusercontent.com/martj42/international-football-results/master/former_names.csv) |
""")

    player_file    = st.file_uploader("① players_fifa25.csv — Player ratings & attributes", type=["csv"])
    team_file      = st.file_uploader("② team_data.csv — Team-level features (ranking, form, host…)", type=["csv"])
    results_file   = st.file_uploader("③ results.csv — International match history 1872-2025", type=["csv"])
    shootouts_file = st.file_uploader("④ shootouts.csv — Penalty shootout history (optional)", type=["csv"])
    fnames_file    = st.file_uploader("⑤ former_names.csv — Country name mapping (optional)", type=["csv"])

    use_demo = st.checkbox(
        "Use synthetic demo data (no upload needed)",
        value=not (player_file and team_file),
        key="use_demo",
    )

    st.divider()
    st.markdown("## ⚙️ Tournament Settings")
    advance_per_group = st.selectbox("Teams advancing per group", [2, 3], index=0,
                                      key="sb_advance")
    include_thirds    = st.checkbox("Include best 8 third-placed teams (2026 rule)", value=True,
                                     key="cb_thirds")
    n_sims_tourney    = st.slider("Monte Carlo simulations", 200, 5000, 1000, step=200,
                                   key="sl_nsims_t")
    elo_years         = st.slider("Elo: use last N years of history (0 = all)", 0, 50, 0, step=5,
                                   key="sl_elo_years")

    st.divider()
    st.markdown("## ⚖️ Strength-Index Weights")
    st.caption("Set to 0 to ignore a feature entirely.")
    weights = {}
    for feat, (inv, def_w, grp, label) in FEATURE_SPECS.items():
        weights[feat] = st.slider(label, 0, 50, def_w, key=f"w_{feat}")


# ═════════════════════════════════════════════════════════════════════════════
# LOAD / BUILD DATA
# ═════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading data…"):

    # ── former-name map ───────────────────────────────────────────────────
    name_map: dict[str, str] = {}
    if fnames_file:
        fn_df = read_csv(fnames_file)
        for _, r in fn_df.iterrows():
            name_map[str(r.get("former", r.get("old_name", ""))).strip()] = \
                str(r.get("current", r.get("new_name", ""))).strip()

    # ── historical Elo ────────────────────────────────────────────────────
    elo_df       = None
    shootouts_df = None
    hist         = None

    if results_file:
        hist = read_csv(results_file)
        hist["home_team"] = hist["home_team"].apply(lambda x: resolve_name(x, name_map))
        hist["away_team"] = hist["away_team"].apply(lambda x: resolve_name(x, name_map))
        elo_df = compute_elo(hist.to_json(), elo_years)
        st.sidebar.success(f"✅ Elo from {len(hist):,} matches · {elo_df['team'].nunique()} teams")

    if shootouts_file:
        shootouts_df = read_csv(shootouts_file)
        for col in ["home_team", "away_team", "winner"]:
            if col in shootouts_df.columns:
                shootouts_df[col] = shootouts_df[col].apply(lambda x: resolve_name(x, name_map))

    # ── player + team ─────────────────────────────────────────────────────
    if use_demo or not (player_file and team_file):
        if not (player_file and team_file):
            st.sidebar.info("📊 Showing synthetic demo data — upload files to use real data.")
        player_df     = make_demo_player_data(ALL_WC_2026_TEAMS)
        team_df       = make_demo_team_data(ALL_WC_2026_TEAMS)
        groups_to_use = WC_2026_GROUPS
    else:
        player_df = read_csv(player_file)
        team_df   = read_csv(team_file)
        if "wc_2026_group" in team_df.columns:
            groups_to_use = {
                str(g): sub["team"].tolist()
                for g, sub in team_df.groupby("wc_2026_group")
            }
        else:
            groups_to_use = WC_2026_GROUPS

    # ── aggregate + merge ─────────────────────────────────────────────────
    squad_df = aggregate_squad_quality(player_df)
    teams_df = pd.merge(team_df, squad_df, on="team", how="outer").dropna(subset=["team"])

    if elo_df is not None:
        teams_df = pd.merge(teams_df, elo_df[["team", "elo_rating"]], on="team", how="left")
        n_miss = teams_df["elo_rating"].isna().sum()
        if n_miss:
            missed = teams_df.loc[teams_df["elo_rating"].isna(), "team"].tolist()
            st.sidebar.warning(f"⚠️ {n_miss} team(s) not in Elo history: {', '.join(missed[:10])}")
            teams_df["elo_rating"] = teams_df["elo_rating"].fillna(teams_df["elo_rating"].mean())
    else:
        if "fifa_points" in teams_df.columns:
            teams_df["elo_rating"] = teams_df["fifa_points"]
        elif "fifa_ranking" in teams_df.columns:
            mr = teams_df["fifa_ranking"].max()
            teams_df["elo_rating"] = (mr - teams_df["fifa_ranking"] + 1) / mr * 1000 + 1000

    teams_df  = build_strength_index(teams_df, weights).reset_index(drop=True)
    teams_list = teams_df["team"].tolist()

    hosts = teams_df.loc[teams_df.get("host_status", pd.Series(0, index=teams_df.index)) == 1, "team"].tolist() \
        if "host_status" in teams_df.columns else []
    host_team = hosts[0] if hosts else None

    # ── fix session_state defaults that reference teams_list ─────────────
    if st.session_state["match_team_a"] not in teams_list:
        st.session_state["match_team_a"] = teams_list[0]
    if st.session_state["match_team_b"] not in teams_list:
        st.session_state["match_team_b"] = teams_list[min(1, len(teams_list) - 1)]
    if not any(t in teams_list for t in st.session_state["tourney_selected"]):
        st.session_state["tourney_selected"] = [t for t in ALL_WC_2026_TEAMS if t in teams_list]


# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.title("🏆 FIFA World Cup 2026 — Prediction Dashboard")
st.caption("Elo-calibrated Poisson model · Dixon-Coles correction · Monte Carlo tournament simulation")

top3 = teams_df.head(3)
c1, c2, c3, c4 = st.columns(4)
c1.metric("🥇 Favourite",  top3.iloc[0]["team"], f"SI {top3.iloc[0]['strength_index']:.1f}")
c2.metric("🥈 Contender",  top3.iloc[1]["team"], f"SI {top3.iloc[1]['strength_index']:.1f}")
c3.metric("🥉 Dark Horse", top3.iloc[2]["team"], f"SI {top3.iloc[2]['strength_index']:.1f}")
c4.metric("Teams Loaded",  str(len(teams_df)),   f"{len(groups_to_use)} groups")
st.divider()


# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_rankings, tab_groups, tab_match, tab_tourney, tab_backtest, tab_data = st.tabs([
    "📊 Rankings", "🗂️ Group Stage", "⚔️ Match Predictor",
    "🏆 Tournament Sim", "✅ Backtest", "🔍 Raw Data",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RANKINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_rankings:
    st.subheader("Team Strength Index — Top 20")

    top20 = teams_df.head(20).copy()
    fig = px.bar(top20, x="strength_index", y="team", orientation="h",
                 color="strength_index", color_continuous_scale="YlOrRd",
                 text="strength_index", template=PLOTLY_TEMPLATE,
                 title="Strength Index — Top 20 Teams")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(style_fig(fig), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.scatter(teams_df, x="attack_strength", y="defense_strength",
                          text="team", color="strength_index",
                          color_continuous_scale="Plasma", template=PLOTLY_TEMPLATE,
                          title="Attack vs Defense Strength")
        fig2.update_traces(textposition="top center", marker=dict(size=10))
        st.plotly_chart(style_fig(fig2), use_container_width=True)
    with col2:
        x_col = "elo_rating" if "elo_rating" in teams_df.columns else "fifa_ranking"
        x_lbl = "Elo Rating" if x_col == "elo_rating" else "FIFA Ranking"
        fig3 = px.scatter(teams_df, x=x_col, y="strength_index", text="team",
                          template=PLOTLY_TEMPLATE,
                          title=f"{x_lbl} vs Composite Strength Index",
                          labels={x_col: x_lbl})
        fig3.update_traces(textposition="top center", marker=dict(size=10))
        st.plotly_chart(style_fig(fig3), use_container_width=True)

    # Radar chart — top 5
    st.subheader("Top-5 Teams — Feature Radar")
    radar_feats = ["attack_strength", "defense_strength", "strength_index"] + \
                  [f for f in ["avg_overall", "avg_passing", "gk_quality"] if f in teams_df.columns]
    radar_fig = go.Figure()
    for _, row in teams_df.head(5).iterrows():
        vals = [float(row.get(f, 50)) for f in radar_feats] + [float(row.get(radar_feats[0], 50))]
        radar_fig.add_trace(go.Scatterpolar(
            r=vals, theta=radar_feats + [radar_feats[0]],
            fill="toself", name=row["team"], opacity=0.75,
        ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template=PLOTLY_TEMPLATE, paper_bgcolor=PAPER_BG, title="Top-5 Feature Radar",
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    with st.expander("📋 Full team table"):
        dc = ["team", "strength_index", "attack_strength", "defense_strength"] + \
             [f for f in FEATURE_SPECS if f in teams_df.columns]
        st.dataframe(teams_df[dc].round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GROUP STAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_groups:
    st.subheader("🗂️ 2026 World Cup Groups")
    grp_items = list(groups_to_use.items())
    for row_start in range(0, len(grp_items), 3):
        cols = st.columns(3)
        for ci, (gname, gteams) in enumerate(grp_items[row_start:row_start + 3]):
            gdf = teams_df[teams_df["team"].isin(gteams)].sort_values("strength_index", ascending=False)
            with cols[ci]:
                st.markdown(f"**Group {gname}**")
                if gdf.empty:
                    st.caption(", ".join(gteams) + " _(not in dataset)_")
                else:
                    gfig = px.bar(gdf, x="team", y="strength_index",
                                  color="strength_index", color_continuous_scale="YlOrRd",
                                  template=PLOTLY_TEMPLATE, height=220, text="strength_index")
                    gfig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
                    gfig.update_layout(showlegend=False, margin=dict(t=10, b=30, l=10, r=10),
                                       paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                                       yaxis_title="", xaxis_title="",
                                       font=dict(color="#e8f5e9", size=11))
                    st.plotly_chart(gfig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MATCH PREDICTOR
# All dropdowns keyed to session_state → they survive reruns without resetting.
# ══════════════════════════════════════════════════════════════════════════════
with tab_match:
    st.subheader("⚔️ Head-to-Head Match Predictor")
    st.markdown(
        "<p style='font-style: italic; font-size: 1.05rem; color:#3ddc84; "
        "margin-top: -8px; margin-bottom: 18px;'>Arun's FIFA 2026 Predictions</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([2, 1, 2])

    # ── Team A selectbox ──────────────────────────────────────────────────
    # index derived from current session_state value so it's stable
    idx_a = teams_list.index(st.session_state["match_team_a"]) \
        if st.session_state["match_team_a"] in teams_list else 0

    team_a = col1.selectbox(
        "Team A", teams_list, index=idx_a,
        key="match_team_a",          # ← session_state key: persists across reruns
    )

    col2.markdown(
        "<br><h2 style='text-align:center;color:#f5c518'>vs</h2>",
        unsafe_allow_html=True,
    )

    # ── Team B selectbox ──────────────────────────────────────────────────
    idx_b = teams_list.index(st.session_state["match_team_b"]) \
        if st.session_state["match_team_b"] in teams_list else min(1, len(teams_list) - 1)

    team_b = col3.selectbox(
        "Team B", teams_list, index=idx_b,
        key="match_team_b",          # ← session_state key
    )

    # ── Other match settings ──────────────────────────────────────────────
    c1, c2 = st.columns(2)
    n_sims_match = c1.slider(
        "Simulations", 500, 10_000, 5_000, step=500,
        key="match_n_sims",
    )
    is_knockout = c2.checkbox(
        "Knockout match (no draws — goes to penalties if level)",
        value=False,
        key="match_knockout",
    )

    # ── Predict button ────────────────────────────────────────────────────
    if st.button("🔮 Predict Outcome", use_container_width=True, key="btn_predict"):
        if team_a == team_b:
            st.warning("Please select two different teams.")
        else:
            with st.spinner("Running simulations…"):
                pa, pdraw, pb = match_outcome_probs(
                    team_a, team_b, teams_df, n_sims_match, host_team=host_team
                )
            # Store result in session_state so it survives the next rerun
            st.session_state["match_result"] = {
                "team_a": team_a, "team_b": team_b,
                "pa": pa, "pdraw": pdraw, "pb": pb,
            }

    # ── Display result (persists between reruns) ──────────────────────────
    if st.session_state["match_result"]:
        r = st.session_state["match_result"]
        pa, pdraw, pb = r["pa"], r["pdraw"], r["pb"]
        ta, tb = r["team_a"], r["team_b"]

        ca, cb, cc = st.columns(3)
        ca.metric(f"🔵 {ta} Win", f"{pa*100:.1f}%")
        cb.metric("⚪ Draw",       f"{pdraw*100:.1f}%")
        cc.metric(f"🔴 {tb} Win", f"{pb*100:.1f}%")

        bar_fig = go.Figure(go.Bar(
            x=[ta, "Draw", tb],
            y=[pa * 100, pdraw * 100, pb * 100],
            marker_color=["#1565c0", "#8fac92", "#e53935"],
            text=[f"{v:.1f}%" for v in [pa*100, pdraw*100, pb*100]],
            textposition="outside",
        ))
        bar_fig.update_layout(
            title=f"{ta} vs {tb} — Outcome Probabilities",
            yaxis_title="Probability (%)", yaxis_range=[0, 100],
            template=PLOTLY_TEMPLATE, paper_bgcolor=PAPER_BG,
            font=dict(color="#e8f5e9"),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # Expected goals + heatmap
        ra_row = teams_df.loc[teams_df.team == ta].iloc[0]
        rb_row = teams_df.loc[teams_df.team == tb].iloc[0]
        la = expected_goals(ra_row["attack_strength"], rb_row["defense_strength"])
        lb = expected_goals(rb_row["attack_strength"], ra_row["defense_strength"])
        st.info(f"📐 Expected goals: **{ta}** {la:.2f}  —  {lb:.2f} **{tb}**  "
                f"(Dixon-Coles corrected Poisson model)")

        MG = 6
        pm = np.zeros((MG, MG))
        for i in range(MG):
            for j in range(MG):
                pm[i, j] = poisson.pmf(i, la) * poisson.pmf(j, lb) * dc_correction(i, j, la, lb)
        pm /= pm.sum()
        hm = px.imshow(
            pm * 100,
            labels=dict(x=f"{tb} goals", y=f"{ta} goals", color="Prob %"),
            x=[str(i) for i in range(MG)], y=[str(i) for i in range(MG)],
            color_continuous_scale="YlOrRd", template=PLOTLY_TEMPLATE,
            title="Score Probability Heatmap (%)", text_auto=".1f",
        )
        hm.update_layout(paper_bgcolor=PAPER_BG, font=dict(color="#e8f5e9"))
        st.plotly_chart(hm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TOURNAMENT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_tourney:
    st.subheader("🏆 Full Tournament Monte Carlo Simulation")
    st.caption(f"Groups → R32 → R16 → QF → SF → Final · "
               f"{n_sims_tourney:,} simulations per run")

    all_group_teams = [t for g in groups_to_use.values() for t in g if t in teams_list]

    # Multiselect also keyed to session_state
    selected_teams = st.multiselect(
        "Teams in the draw (edit to customise)",
        options=teams_list,
        default=[t for t in st.session_state["tourney_selected"] if t in teams_list] or all_group_teams,
        key="tourney_selected",
    )

    custom_groups = {
        g: [t for t in gteams if t in selected_teams]
        for g, gteams in groups_to_use.items()
    }
    custom_groups = {g: t for g, t in custom_groups.items() if t}

    if st.button("🎲 Run Tournament Simulation", use_container_width=True, key="btn_tourney"):
        valid = [t for t in selected_teams if t in teams_df["team"].values]
        if len(valid) < 4:
            st.error("Select at least 4 teams that exist in the dataset.")
        else:
            with st.spinner(f"Simulating {n_sims_tourney:,} World Cups…"):
                sim_df = teams_df[teams_df["team"].isin(valid)].copy().reset_index(drop=True)
                import json
                groups_json = json.dumps({k: list(v) for k, v in custom_groups.items()})
                result_df = run_monte_carlo(
                    groups_json=groups_json,
                    teams_json=sim_df.to_json(),
                    n_sims=n_sims_tourney,
                    advance_per_group=advance_per_group,
                    host_team=host_team,
                    include_best_thirds=include_thirds,
                )
            st.session_state["tourney_result"] = result_df.to_json()

    if st.session_state["tourney_result"]:
        results = pd.read_json(io.StringIO(st.session_state["tourney_result"]))

        top_n   = min(20, len(results))
        top_res = results.head(top_n)

        wfig = px.bar(top_res, x="win_probability_%", y="team", orientation="h",
                      color="win_probability_%", color_continuous_scale="YlOrRd",
                      text="win_probability_%", template=PLOTLY_TEMPLATE,
                      title=f"Probability of Winning the 2026 World Cup (top {top_n})")
        wfig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        wfig.update_layout(yaxis={"categoryorder": "total ascending"},
                           paper_bgcolor=PAPER_BG, font=dict(color="#e8f5e9"))
        st.plotly_chart(wfig, use_container_width=True)

        top8      = results.head(8)
        rest_pct  = results.iloc[8:]["win_probability_%"].sum()
        pie_fig   = go.Figure(go.Pie(
            labels=top8["team"].tolist() + ["Rest of World"],
            values=top8["win_probability_%"].tolist() + [round(rest_pct, 2)],
            hole=0.4, textinfo="label+percent",
            marker=dict(colors=px.colors.qualitative.Vivid),
        ))
        pie_fig.update_layout(title="Share of wins — top 8 + Rest",
                               template=PLOTLY_TEMPLATE, paper_bgcolor=PAPER_BG,
                               font=dict(color="#e8f5e9"))
        st.plotly_chart(pie_fig, use_container_width=True)

        fav = results.iloc[0]
        st.success(
            f"🏆 **Predicted champion: {fav['team']}** — "
            f"won {fav['simulations_won']} of {n_sims_tourney} simulated tournaments "
            f"({fav['win_probability_%']:.1f}%)"
        )
        with st.expander("📋 Full results table"):
            st.dataframe(results, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.subheader("✅ Backtest Against Real Historical Matches")

    if hist is None:
        st.info("Upload **results.csv** (file ③ above) to enable backtesting.")
    else:
        col1, col2 = st.columns(2)
        n_back   = col1.slider("Matches to test", 20, 1000, 200, step=20, key="sl_nback")
        min_year = col2.slider("From year", 1990, 2024, 2010, key="sl_minyear")

        if st.button("▶ Run Backtest", use_container_width=True, key="btn_backtest"):
            teams_set = set(teams_df["team"].tolist())
            hist_copy = hist.copy()
            hist_copy["date"] = pd.to_datetime(hist_copy["date"], errors="coerce")
            recent = (
                hist_copy
                .dropna(subset=["home_score", "away_score", "date"])
                .query("date.dt.year >= @min_year")
                .pipe(lambda d: d[d.home_team.isin(teams_set) & d.away_team.isin(teams_set)])
                .sort_values("date", ascending=False)
                .head(n_back)
            )
            rows = []
            for _, m in recent.iterrows():
                h, a   = m["home_team"], m["away_team"]
                hs, as_ = float(m["home_score"]), float(m["away_score"])
                actual  = h if hs > as_ else (a if as_ > hs else "Draw")
                rh = teams_df.loc[teams_df.team == h]
                ra = teams_df.loc[teams_df.team == a]
                if rh.empty or ra.empty: continue
                rh, ra = rh.iloc[0], ra.iloc[0]
                lh = expected_goals(rh["attack_strength"], ra["defense_strength"])
                la = expected_goals(ra["attack_strength"], rh["defense_strength"])
                pred = h if lh > la else (a if la > lh else "Draw")
                rows.append({
                    "date": m["date"].date(), "home": h, "away": a,
                    "score": f"{int(hs)}-{int(as_)}", "actual_winner": actual,
                    "model_favourite": pred, "xG_home": round(lh, 2),
                    "xG_away": round(la, 2), "correct": (pred == actual),
                    "tournament": m.get("tournament", ""),
                })
            if not rows:
                st.warning("No overlapping matches found — check team-name spelling.")
            else:
                bt  = pd.DataFrame(rows)
                acc = bt["correct"].mean() * 100
                c1, c2, c3 = st.columns(3)
                c1.metric("Favourite-pick accuracy", f"{acc:.1f}%")
                c2.metric("Matches evaluated", str(len(bt)))
                c3.metric("Correct calls", str(bt["correct"].sum()))
                st.caption("📌 Theoretical ceiling for 3-outcome football prediction ≈ 55–60%.")
                if bt["tournament"].nunique() > 1:
                    by_t = (
                        bt.groupby("tournament")["correct"]
                        .agg(["mean", "count"]).rename(columns={"mean":"acc","count":"n"})
                        .sort_values("acc", ascending=False).head(12).reset_index()
                    )
                    by_t["acc_%"] = (by_t["acc"] * 100).round(1)
                    bfig = px.bar(by_t, x="acc_%", y="tournament", orientation="h",
                                  text="acc_%", template=PLOTLY_TEMPLATE,
                                  title="Accuracy by Tournament Type")
                    bfig.update_layout(paper_bgcolor=PAPER_BG,
                                       yaxis={"categoryorder":"total ascending"},
                                       font=dict(color="#e8f5e9"))
                    st.plotly_chart(bfig, use_container_width=True)
                st.dataframe(bt, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("🔍 Underlying Data")
    with st.expander("Squad quality (aggregated from player file)"):
        st.dataframe(squad_df.round(3), use_container_width=True)
    with st.expander("Team-level file (raw upload / demo)"):
        st.dataframe(team_df, use_container_width=True)
    with st.expander("Merged + strength-indexed team table"):
        st.dataframe(teams_df.round(3), use_container_width=True)
    if elo_df is not None:
        with st.expander("Elo leaderboard (top 30)"):
            st.dataframe(elo_df.head(30).round(1), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "v4 · Fixed official 48-team / 12-group draw (Algeria, DR Congo, Cape Verde, "
    "Curaçao, Jordan, Uzbekistan, Iraq, Ivory Coast, Ghana, etc. now included) · "
    "Dropdown-reset bug fixed (session_state) · Elo K-factor capped · "
    "Dixon-Coles low-score correction · Attack/Defense split Poisson model · "
    "Monte Carlo tournament sim | Built with Streamlit + NumPy + SciPy + Plotly"
)
