import numpy as np
import pandas as pd
import pytest

from src.xpts_engine import (
    build_fixture_expectations, compute_xpts_matrix,
    expected_goals_conceded_penalty, league_goals_per_team_match, _shrunk_per90,
)

TEAM_MAP = {1: "A", 2: "B", 3: "C"}


def _fx(rows):
    base = pd.Timestamp("2026-11-07T15:00:00Z")
    return pd.DataFrame([{
        "event": gw, "team_h": h, "team_a": a, "finished": False,
        "kickoff_time": (base + pd.Timedelta(days=7 * i)).isoformat(),
    } for i, (gw, h, a) in enumerate(rows)])


def _flat():
    return pd.DataFrame({"Team": ["A", "B", "C"], "Attack_Strength": [1.0] * 3, "Defense_Vulnerability": [1.0] * 3})


def _player(pos="FWD", minutes=540, xg=0.0, xa=0.0, chance=None, team="A", status="a"):
    return {"Team": team, "Position": pos, "Minutes": minutes, "xG": xg, "xA": xa,
            "Chance_of_Playing": chance, "status": status}


def test_blank_and_double_gameweeks():
    # A: double in GW7 (two home games), blank in GW8, single in GW9
    fx = _fx([(7, 1, 2), (7, 1, 3), (9, 1, 2), (8, 2, 3)])
    table = build_fixture_expectations(fx, _flat(), TEAM_MAP)
    players = pd.DataFrame([_player()])   # FWD, no attacking output: only appearance points
    mat, gws = compute_xpts_matrix(players, table, {"A": 6, "B": 6, "C": 6}, n_gws=3)

    assert gws == [7, 8, 9]
    assert mat.loc[0, "xPts_GW2"] == 0.0                              # blank
    assert mat.loc[0, "xPts_GW3"] == pytest.approx(2.0)               # one fixture, 90 min starter
    assert mat.loc[0, "xPts_GW1"] == pytest.approx(2 * mat.loc[0, "xPts_GW3"])  # double


def test_weaker_opponent_means_more_points():
    ratings = pd.DataFrame({"Team": ["A", "B", "C"], "Attack_Strength": [1.0, 1.0, 1.0],
                            "Defense_Vulnerability": [1.0, 0.6, 1.4]})   # B solid, C leaky
    fx = _fx([(7, 1, 2), (8, 1, 3)])
    table = build_fixture_expectations(fx, ratings, TEAM_MAP)
    players = pd.DataFrame([_player("MID", xg=3.0, xa=2.0), _player("DEF", xg=1.0, xa=1.0)])
    mat, _ = compute_xpts_matrix(players, table, {"A": 6, "B": 6, "C": 6}, n_gws=2)
    assert (mat["xPts_GW2"] > mat["xPts_GW1"]).all()     # leaky C beats solid B


def test_clean_sheet_depends_on_opponent_attack_and_own_defence():
    ratings = pd.DataFrame({"Team": ["A", "B", "C"], "Attack_Strength": [1.0, 1.5, 0.6],
                            "Defense_Vulnerability": [1.0, 1.0, 1.0]})
    fx = _fx([(7, 1, 2), (8, 1, 3)])    # GW7 vs potent B, GW8 vs toothless C (both home)
    table = build_fixture_expectations(fx, ratings, TEAM_MAP).set_index(["GW", "Team"])
    assert table.loc[(8, "A"), "P_CS"] > table.loc[(7, "A"), "P_CS"]
    assert table.loc[(8, "A"), "E_GC_Pen"] < table.loc[(7, "A"), "E_GC_Pen"]


def test_availability_and_minutes():
    fx = _fx([(7, 1, 2)])
    table = build_fixture_expectations(fx, _flat(), TEAM_MAP)
    players = pd.DataFrame([
        _player(),                        # healthy 90-minute starter
        _player(chance=50),               # 50% to play
        _player(chance=0),                # ruled out
        _player(status="u"),              # unavailable
        _player(minutes=0),               # never played
    ])
    mat, _ = compute_xpts_matrix(players, table, {"A": 6}, n_gws=1)
    col = mat["xPts_GW1"].tolist()
    assert col[0] == pytest.approx(2.0)
    assert col[1] == pytest.approx(1.0)
    assert col[2] == 0.0 and col[3] == 0.0 and col[4] == 0.0


def test_small_samples_are_shrunk_toward_position_average():
    players = pd.DataFrame([
        {"Position": "MID", "xG": 0.5, "Minutes": 30},     # cameo: raw 1.5 xG/90
        {"Position": "MID", "xG": 2.0, "Minutes": 900},    # regular: 0.2 xG/90
        {"Position": "MID", "xG": 2.0, "Minutes": 900},
    ])
    exposure = players["Minutes"] / 90.0
    rate = _shrunk_per90(players, "xG", exposure, prior_90s=3.0)
    position_avg = 4.5 / (players["Minutes"].sum() / 90.0)      # ~0.22 xG per 90
    assert rate[0] < 0.5                                       # raw cameo rate is 1.5 per 90
    assert abs(rate[0] - position_avg) < abs(rate[0] - 1.5)    # much closer to the average than to the raw rate
    assert rate[1] == pytest.approx(0.2, abs=0.03)             # big sample barely moves


def test_goals_conceded_penalty_matches_brute_force_and_is_monotonic():
    mu = 1.7
    from math import factorial, exp
    brute = sum((k // 2) * exp(-mu) * mu ** k / factorial(k) for k in range(0, 40))
    assert expected_goals_conceded_penalty(mu)[0] == pytest.approx(brute, abs=1e-6)
    assert expected_goals_conceded_penalty(0.0)[0] == 0.0
    vals = expected_goals_conceded_penalty([0.5, 1.0, 2.0, 3.0])
    assert (np.diff(vals) > 0).all()


def test_own_key_absences_reduce_own_attack_multiplier_only():
    fx = _fx([(7, 1, 2)])
    base = build_fixture_expectations(fx, _flat(), TEAM_MAP).set_index("Team")
    hurt = build_fixture_expectations(fx, _flat(), TEAM_MAP, key_absences={"A": 0.2}).set_index("Team")
    assert hurt.loc["A", "Att_Mult"] < base.loc["A", "Att_Mult"]
    assert hurt.loc["B", "Att_Mult"] > base.loc["B", "Att_Mult"]   # A now leaks more


def test_league_goal_average():
    fx = pd.DataFrame({"team_h_score": [2] * 10, "team_a_score": [1] * 10, "finished": [True] * 10})
    assert league_goals_per_team_match(fx) == pytest.approx(1.5)
    assert league_goals_per_team_match(fx.head(3)) == pytest.approx(1.4)   # too few -> default