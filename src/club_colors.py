"""
Club identity colors — used to tint table rows so a player's club is
recognizable at a glance without reading the Team column text every time.

Colors are approximate brand/kit primary colors, not official trademarked
hex values. Coverage includes clubs that commonly cycle between the
Premier League and Championship; any club not listed falls back to a
deterministic (hash-based) color from a neutral palette, so newly
promoted/relegated clubs never break the lookup.
"""
import hashlib

CLUB_COLORS = {
    "Arsenal": "#EF0107",
    "Aston Villa": "#95BFE5",
    "Bournemouth": "#DA291C",
    "Brentford": "#E30613",
    "Brighton": "#0057B8",
    "Burnley": "#6C1D45",
    "Chelsea": "#034694",
    "Crystal Palace": "#1B458F",
    "Everton": "#003399",
    "Fulham": "#000000",
    "Ipswich": "#3A64A3",
    "Leeds": "#FFCD00",
    "Leicester": "#003090",
    "Liverpool": "#C8102E",
    "Luton": "#F78F1E",
    "Man City": "#6CABDD",
    "Man Utd": "#DA291C",
    "Newcastle": "#241F20",
    "Norwich": "#00A650",
    "Nott'm Forest": "#DD0000",
    "Sheffield Utd": "#EE2737",
    "Southampton": "#D71920",
    "Spurs": "#132257",
    "Sunderland": "#EB172B",
    "Watford": "#FBEE23",
    "West Brom": "#122F67",
    "West Ham": "#7A263A",
    "Wolves": "#FDB913",
}

_FALLBACK_PALETTE = [
    "#7f8c8d", "#8e6c88", "#5c8374", "#a67c52", "#6b7fd7", "#c97064",
]


def get_club_color(team_name: str) -> str:
    """Returns a hex color for the given club. Known clubs get their real
    identity color; unknown clubs get a stable (not random-per-render)
    color derived from a hash of the name, so the same club always maps
    to the same fallback color across reruns."""
    if team_name in CLUB_COLORS:
        return CLUB_COLORS[team_name]
    if not team_name:
        return _FALLBACK_PALETTE[0]
    digest = int(hashlib.md5(team_name.encode()).hexdigest(), 16)
    return _FALLBACK_PALETTE[digest % len(_FALLBACK_PALETTE)]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def style_table_by_club(df, team_col: str = "Team", alpha: float = 0.16):
    """Returns a pandas Styler that tints each row with a translucent
    background based on the player's club color, so scanning a long table
    for 'players from Team X' doesn't require reading every row's text."""
    def _row_style(row):
        color = get_club_color(row.get(team_col, ""))
        bg = _hex_to_rgba(color, alpha)
        return [f"background-color: {bg}"] * len(row)

    return df.style.apply(_row_style, axis=1)