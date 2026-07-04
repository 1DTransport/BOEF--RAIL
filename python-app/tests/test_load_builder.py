import pytest

from core.load_builder import (
    AS5100RailLoadConfig,
    as5100_load_metadata,
    build_as5100_group_spacing_candidates,
    build_as5100_rail_loads,
    TrainLoadConfig,
    build_train_loads,
)


def test_build_train_loads_two_bogies_two_axles() -> None:
    config = TrainLoadConfig(
        axle_load_n=100_000.0,
        bogie_count=2,
        bogie_spacing_m=2.5,
        axles_per_bogie=2,
        axle_spacing_m=1.6,
        reference_bogie_center_m=0.0,
    )
    loads = build_train_loads(config)
    positions = [load.position_m for load in loads]
    expected = [-0.8, 0.8, 1.7, 3.3]
    assert positions == pytest.approx(expected)
    assert [load.load_newtons for load in loads] == pytest.approx([50_000.0] * 4)


def test_build_train_loads_single_bogie_multiple_axles() -> None:
    config = TrainLoadConfig(
        axle_load_n=90_000.0,
        bogie_count=1,
        bogie_spacing_m=0.0,
        axles_per_bogie=3,
        axle_spacing_m=1.2,
        reference_bogie_center_m=5.0,
    )
    loads = build_train_loads(config)
    positions = [load.position_m for load in loads]
    expected = [3.8, 5.0, 6.2]
    assert positions == pytest.approx(expected)
    assert [load.load_newtons for load in loads] == pytest.approx([45_000.0] * 3)


def test_build_train_loads_requires_spacing_for_multiple_bogies() -> None:
    config = TrainLoadConfig(
        axle_load_n=80_000.0,
        bogie_count=2,
        bogie_spacing_m=0.0,
        axles_per_bogie=2,
        axle_spacing_m=1.0,
        reference_bogie_center_m=0.0,
    )
    with pytest.raises(ValueError, match="bogie_spacing_m must be positive"):
        build_train_loads(config)


def test_build_as5100_300la_fixed_arrangement() -> None:
    config = AS5100RailLoadConfig(
        model="300LA",
        group_count=2,
        group_spacing_m=12.0,
        reference_position_m=-1.0,
    )

    loads = build_as5100_rail_loads(config)

    assert [load.position_m for load in loads] == pytest.approx(
        [-1.0, 1.0, 2.7, 3.8, 5.5, 13.0, 14.7, 15.8, 17.5]
    )
    assert [load.load_newtons for load in loads] == pytest.approx(
        [180_000.0, 150_000.0, 150_000.0, 150_000.0, 150_000.0, 150_000.0, 150_000.0, 150_000.0, 150_000.0]
    )


def test_build_as5100_150la_is_half_300la() -> None:
    base = AS5100RailLoadConfig(
        model="300LA",
        group_count=2,
        group_spacing_m=20.0,
        reference_position_m=0.0,
    )
    light = AS5100RailLoadConfig(
        model="150LA",
        group_count=2,
        group_spacing_m=20.0,
        reference_position_m=0.0,
    )

    base_loads = build_as5100_rail_loads(base)
    light_loads = build_as5100_rail_loads(light)

    assert [load.position_m for load in light_loads] == pytest.approx(
        [load.position_m for load in base_loads]
    )
    assert [load.load_newtons for load in light_loads] == pytest.approx(
        [0.5 * load.load_newtons for load in base_loads]
    )


def test_build_as5100_validates_group_spacing_and_count() -> None:
    with pytest.raises(ValueError, match="between 12 m and 20 m"):
        build_as5100_rail_loads(
            AS5100RailLoadConfig(model="300LA", group_count=1, group_spacing_m=11.99)
        )
    with pytest.raises(ValueError, match="between 12 m and 20 m"):
        build_as5100_rail_loads(
            AS5100RailLoadConfig(model="300LA", group_count=1, group_spacing_m=20.01)
        )
    with pytest.raises(ValueError, match="group_count must be positive"):
        build_as5100_rail_loads(
            AS5100RailLoadConfig(model="300LA", group_count=0, group_spacing_m=12.0)
        )


def test_as5100_metadata_is_traceable_and_json_safe() -> None:
    config = AS5100RailLoadConfig(
        model="300LA",
        group_count=1,
        group_spacing_m=12.0,
        reference_position_m=0.0,
    )
    loads = build_as5100_rail_loads(config)

    metadata = as5100_load_metadata(config, loads=loads)

    assert metadata["source_type"] == "as5100_fixed_rail"
    assert metadata["standard"] == "AS5100.2:2017"
    assert metadata["model"] == "300LA"
    assert metadata["arrangement"] == "fixed_user_selected"
    assert metadata["load_basis"] == "axle_load_split_to_two_rails"
    assert metadata["solver_load_basis"] == "wheel_load_per_rail"
    assert metadata["axle_count"] == 5
    assert metadata["max_axle_load_n"] == pytest.approx(360_000.0)
    assert metadata["max_wheel_load_n_per_rail"] == pytest.approx(180_000.0)
    assert metadata["axle_loads_n"] == pytest.approx(
        [360_000.0, 300_000.0, 300_000.0, 300_000.0, 300_000.0]
    )
    assert metadata["wheel_loads_n_per_rail"] == pytest.approx(
        [180_000.0, 150_000.0, 150_000.0, 150_000.0, 150_000.0]
    )
    assert metadata["automatic_dla_applied"] is False


def test_as5100_group_spacing_candidates_include_bounds_and_selected() -> None:
    assert build_as5100_group_spacing_candidates(12.0) == [12.0, 20.0]
    assert build_as5100_group_spacing_candidates(16.5) == [12.0, 16.5, 20.0]
