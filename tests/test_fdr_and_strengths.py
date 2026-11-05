import pandas as pd
import pytest

from src.custom_xpts import build_custom_fdr_matrix
from src.predictor import compute_team_strengths_from_results


def _teams():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Strong", "Mid", "Leaky", "NoGames"],
        "short_name": ["STR", "MID", "LEK", "NOG"],
        "strength_attack_home": [1200] * 4,
        "strength_attack_away": [1200] * 4,
        "strength_defence_home": [1200] * 4,
        "strength_defence_away": [1200] * 4,
    })


def _finished_fixtures():
    # Strong 3-0 Mid | Leaky 0-3 Strong | Mid 2-0 Leaky  (NoGames has not played)
    return pd.DataFrame({
        "team_h": [1, 3, 2],
        "team_a": [2, 1, 3],
        "team_h_score": [3, 0, 2],
        "team_a_score": [0, 3, 0],
        "finished": [True, True, True],
    })


def test_strengths_follow_results_and_average_to_one():
    out = compute_team_strengths_from_results(_finished_fixtures(), _teams()).set_index("Team")

    assert out["Attack_Strength"].mean() == pytest.approx(1.0)
    assert out["Defense_Vulnerability"].mean() == pytest.approx(1.0)
    assert out.notna().all().all()

    assert out.loc["Strong", "Attack_Strength"] > out.loc["Mid", "Attack_Strength"] > out.loc["Leaky", "Attack_Strength"]
    assert out.loc["Leaky", "Defense_Vulnerability"] > out.loc["Mid", "Defense_Vulnerability"] > out.loc["Strong", "Defense_Vulnerability"]

    # A team with no matches sits between the best and worst, never NaN/extreme
    assert out.loc["Leaky", "Attack_Strength"] < out.loc["NoGames", "Attack_Strength"] < out.loc["Strong", "Attack_Strength"]
    assert out.loc["NoGames", "Matches_Played"] == 0


def test_no_results_falls_back_to_fpl_prior():
    out = compute_team_strengths_from_results(pd.DataFrame(), _teams())
    assert len(out) == 4
    assert out["Attack_Strength"].tolist() == pytest.approx([1.0] * 4)
    assert out["Defense_Vulnerability"].tolist() == pytest.approx([1.0] * 4)


def _upcoming():
    return pd.DataFrame({
        "event": [1, 2],
        "team_h": [2, 2],
        "team_a": [1, 3],  # GW1: Mid v Strong, GW2: Mid v Leaky
        "finished": [False, False],
        "kickoff_time": ["2026-11-07T15:00:00Z", "2026-11-14T15:00:00Z"],
    })


def _ratings(mid_attack=1.0):
    return pd.DataFrame({
        "Team": ["Strong", "Mid", "Leaky"],
        "Attack_Strength": [1.4, mid_attack, 0.8],
        "Defense_Vulnerability": [0.6, 1.0, 1.4],
    })


TEAM_MAP = {1: "Strong", 2: "Mid", 3: "Leaky"}
SHORT = {"Strong": "STR", "Mid": "MID", "Leaky": "LEK"}


def test_fdr_direction_is_correct():
    res = build_custom_fdr_matrix(_upcoming(), _ratings(), TEAM_MAP, short_names=SHORT)
    atk_lbl, atk_val = res["attack"]
    def_lbl, def_val = res["defense"]

    # Scoring vs a solid defence (Strong) is HARDER than vs a leaky one (Leaky)
    assert atk_val.loc["Mid", "GW1"] > atk_val.loc["Mid", "GW2"]
    # Keeping a clean sheet vs a potent attack (Strong) is HARDER than vs Leaky
    assert def_val.loc["Mid", "GW1"] > def_val.loc["Mid", "GW2"]
    assert atk_lbl.loc["Mid", "GW1"] == "STR (H)"


def test_fdr_ignores_own_team_strength():
    base = build_custom_fdr_matrix(_upcoming(), _ratings(1.0), TEAM_MAP, short_names=SHORT)
    boosted = build_custom_fdr_matrix(_upcoming(), _ratings(2.0), TEAM_MAP, short_names=SHORT)
    for key in ("attack", "defense"):
        pd.testing.assert_series_equal(base[key][1].loc["Mid"], boosted[key][1].loc["Mid"])


def test_home_is_easier_than_away():
    flat = pd.DataFrame({
        "Team": ["Strong", "Mid", "Leaky"],
        "Attack_Strength": [1.0, 1.0, 1.0],
        "Defense_Vulnerability": [1.0, 1.0, 1.0],
    })
    res = build_custom_fdr_matrix(_upcoming(), flat, TEAM_MAP, short_names=SHORT)
    _, atk_val = res["attack"]
    # Average opponent: 3.5 +/- the venue delta (defaults -0.25 home / +0.25 away)
    assert atk_val.loc["Mid", "GW1"] == pytest.approx(3.25)     # Mid at home
    assert atk_val.loc["Strong", "GW1"] == pytest.approx(3.75)  # Strong away

@pytest.mark.parametrize("bad_value", [0, None, float("nan")])
def test_invalid_fpl_ratings_never_produce_nan(bad_value):
    teams = _teams()
    for col in ("strength_attack_home", "strength_attack_away",
                "strength_defence_home", "strength_defence_away"):
        teams[col] = bad_value
    out = compute_team_strengths_from_results(_finished_fixtures(), teams)
    assert out[["Attack_Strength", "Defense_Vulnerability"]].notna().all().all()
    assert out["Attack_Strength"].mean() == pytest.approx(1.0)
    # results still drive the ranking when the prior is unusable
    out = out.set_index("Team")
    assert out.loc["Strong", "Attack_Strength"] > out.loc["Leaky", "Attack_Strength"]


def test_partially_missing_fpl_ratings_use_median():
    teams = _teams()
    teams.loc[3, "strength_attack_home"] = float("nan")  # NoGames: one value missing
    teams.loc[3, "strength_defence_home"] = 0
    out = compute_team_strengths_from_results(_finished_fixtures(), teams)
    assert out[["Attack_Strength", "Defense_Vulnerability"]].notna().all().all()