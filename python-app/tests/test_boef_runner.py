from pathlib import Path


def test_boef_runner_has_expected_commands() -> None:
    runner_path = Path(__file__).resolve().parents[1] / "boef"
    contents = runner_path.read_text(encoding="utf-8")

    assert "run)" in contents
    assert "core)" in contents
    assert "db)" in contents
    assert "test)" in contents


def test_boef_runner_avoids_build_isolation() -> None:
    runner_path = Path(__file__).resolve().parents[1] / "boef"
    contents = runner_path.read_text(encoding="utf-8")

    assert "--no-build-isolation" in contents
