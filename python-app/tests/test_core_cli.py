from __future__ import annotations

import json
import sys

from core.cli import main


def test_core_cli_legacy_deflection_mode(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "core",
            "--load-newtons",
            "10000",
            "--stiffness-newtons-per-meter",
            "2000000",
        ],
    )
    assert main() == 0
    captured = capsys.readouterr()
    assert "Deflection: 0.005000 m" in captured.out


def test_core_cli_deflection_subcommand_json(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "core",
            "deflection",
            "--load-newtons",
            "10000",
            "--stiffness-newtons-per-meter",
            "2000000",
            "--json",
        ],
    )
    assert main() == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out.strip())
    assert parsed["deflection_m"] == 0.005


def test_core_cli_a3902_json_track_class(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "core",
            "a3902",
            "--track-class",
            "3",
            "--static-wheel-load-n",
            "100000",
            "--speed-kmh",
            "80",
            "--confidence-limit-tc",
            "1.0",
            "--beta-per-m",
            "0.9",
            "--foundation-modulus-n-per-m2",
            "80000000",
            "--sleeper-spacing-m",
            "0.65",
            "--sleeper-width-m",
            "0.25",
            "--sleeper-length-m",
            "2.5",
            "--rail-centres-m",
            "1.5",
            "--factor-of-safety-f1",
            "1.25",
            "--factor-of-safety-f2",
            "1.0",
            "--ballast-depth-m",
            "0.3",
            "--fill-depth-m",
            "0.2",
            "--json",
        ],
    )
    assert main() == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out.strip())
    assert parsed["vqi"] == 65.0
    assert parsed["p_dv_n"] > parsed["p_sv_n"]
    assert parsed["p_a_pa"] > 0.0
    assert parsed["p_f_pa"] is not None
    assert parsed["p_s_pa"] is not None


def test_core_cli_a3902_zero_ballast_depth_skips_pf_ps(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "core",
            "a3902",
            "--track-class",
            "3",
            "--static-wheel-load-n",
            "100000",
            "--speed-kmh",
            "80",
            "--confidence-limit-tc",
            "1.0",
            "--beta-per-m",
            "0.9",
            "--foundation-modulus-n-per-m2",
            "80000000",
            "--sleeper-spacing-m",
            "0.65",
            "--sleeper-width-m",
            "0.25",
            "--sleeper-length-m",
            "2.5",
            "--rail-centres-m",
            "1.5",
            "--ballast-depth-m",
            "0",
            "--fill-depth-m",
            "0.2",
            "--json",
        ],
    )
    assert main() == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out.strip())
    assert parsed["p_f_pa"] is None
    assert parsed["p_s_pa"] is None
