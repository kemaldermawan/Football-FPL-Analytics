"""
Central configuration: filesystem paths, theme colors, and shared constants.
Keeping these in one place avoids magic strings scattered across modules.
"""
import os
"""
Central configuration: filesystem paths, theme colors, and shared constants.
Keeping these in one place avoids magic strings scattered across modules.
"""
import os

# --- Filesystem ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

PATH_FPL_STATIC = os.path.join(DATA_DIR, "fpl_static.parquet")
PATH_FPL_FIXTURES = os.path.join(DATA_DIR, "fpl_fixtures.parquet")
PATH_FPL_TEAMS = os.path.join(DATA_DIR, "fpl_teams.parquet")
PATH_NEXT_OPPONENT = os.path.join(DATA_DIR, "team_next_opponent.parquet")
PATH_FIXTURE_RUN = os.path.join(DATA_DIR, "team_fixture_run.parquet")
PATH_SEASON_STATUS = os.path.join(DATA_DIR, "season_status.json")
PATH_EVENT_RAW = os.path.join(DATA_DIR, "epl_raw.parquet")
PATH_ADVANCED_STATS = os.path.join(DATA_DIR, "advanced_fbref_stats.parquet")
PATH_TEAM_FORM = os.path.join(DATA_DIR, "team_rolling_form.parquet")

# --- FPL API ---
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_ENTRY_PICKS_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/picks/"

# --- Theme: neutral dark "analytics desk" aesthetic used across the
# Streamlit chrome and matplotlib/altair renders. Green/red are reserved
# for data semantics (goal vs miss, positive vs negative delta) rather
# than general UI chrome, so the dashboard doesn't read as monochrome.
COLOR_BG = "#12151c"            # app background — neutral dark slate
COLOR_PANEL = "#1a1e28"         # sidebar / card background
COLOR_PITCH = "#1d2129"         # matplotlib figure background (pitch charts)
COLOR_LINE = "#5b6472"          # gridlines / pitch lines
COLOR_ACCENT = "#4f8cf0"        # primary UI accent (buttons, active tab, headers)
COLOR_ACCENT_SOFT = "#8fb3f2"   # secondary accent (bars, non-semantic highlights)
COLOR_POSITIVE = "#3fb27f"      # semantic: goals, gains, positive deltas
COLOR_NEGATIVE = "#e2685f"      # semantic: misses, losses, negative deltas
COLOR_GOLD = "#e0b04f"          # semantic: warnings / captaincy / highlights
COLOR_TEXT = "#e8eaf0"
COLOR_MUTED = "#8b93a3"

# Backward-compatible aliases (older modules may still import these names)
COLOR_ACCENT_GREEN = COLOR_POSITIVE
COLOR_ACCENT_RED = COLOR_NEGATIVE

POSITION_ORDER = ["GKP", "DEF", "MID", "FWD"]
POSITION_LIMITS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_XI_LIMITS = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
MAX_PLAYERS_PER_CLUB = 3
SQUAD_SIZE = 15
BUDGET_DEFAULT = 100.0
# --- Filesystem ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

PATH_FPL_STATIC = os.path.join(DATA_DIR, "fpl_static.parquet")
PATH_FPL_FIXTURES = os.path.join(DATA_DIR, "fpl_fixtures.parquet")
PATH_FPL_TEAMS = os.path.join(DATA_DIR, "fpl_teams.parquet")
PATH_NEXT_OPPONENT = os.path.join(DATA_DIR, "team_next_opponent.parquet")
PATH_SEASON_STATUS = os.path.join(DATA_DIR, "season_status.json")
PATH_EVENT_RAW = os.path.join(DATA_DIR, "epl_raw.parquet")
PATH_ADVANCED_STATS = os.path.join(DATA_DIR, "advanced_fbref_stats.parquet")
PATH_TEAM_FORM = os.path.join(DATA_DIR, "team_rolling_form.parquet")

# --- FPL API ---
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_ENTRY_PICKS_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/picks/"

# --- Theme: neutral dark "analytics desk" aesthetic used across the
# Streamlit chrome and matplotlib/altair renders. Green/red are reserved
# for data semantics (goal vs miss, positive vs negative delta) rather
# than general UI chrome, so the dashboard doesn't read as monochrome.
COLOR_BG = "#12151c"            # app background — neutral dark slate
COLOR_PANEL = "#1a1e28"         # sidebar / card background
COLOR_PITCH = "#1d2129"         # matplotlib figure background (pitch charts)
COLOR_LINE = "#5b6472"          # gridlines / pitch lines
COLOR_ACCENT = "#4f8cf0"        # primary UI accent (buttons, active tab, headers)
COLOR_ACCENT_SOFT = "#8fb3f2"   # secondary accent (bars, non-semantic highlights)
COLOR_POSITIVE = "#3fb27f"      # semantic: goals, gains, positive deltas
COLOR_NEGATIVE = "#e2685f"      # semantic: misses, losses, negative deltas
COLOR_GOLD = "#e0b04f"          # semantic: warnings / captaincy / highlights
COLOR_TEXT = "#e8eaf0"
COLOR_MUTED = "#8b93a3"

# Backward-compatible aliases (older modules may still import these names)
COLOR_ACCENT_GREEN = COLOR_POSITIVE
COLOR_ACCENT_RED = COLOR_NEGATIVE

POSITION_ORDER = ["GKP", "DEF", "MID", "FWD"]
POSITION_LIMITS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_XI_LIMITS = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
MAX_PLAYERS_PER_CLUB = 3
SQUAD_SIZE = 15
BUDGET_DEFAULT = 100.0