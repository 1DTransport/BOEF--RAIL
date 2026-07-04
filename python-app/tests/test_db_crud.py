import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from db import crud
from db.models import Base, DesignAlternative, Rail
from db.seed import seed_database


def _make_session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def test_crud_round_trip_for_track_config() -> None:
    with _make_session() as session:
        rail = crud.create_rail(
            session,
            name="Test Rail",
            elastic_modulus_pa=2.1e11,
            moment_inertia_m4=3.05e-5,
            section_modulus_m3=4.1e-4,
            mass_kg_per_m=60.0,
        )
        sleeper = crud.create_sleeper(
            session,
            name="Test Sleeper",
            elastic_modulus_pa=3.2e10,
            length_m=2.5,
            width_m=0.25,
            height_m=0.21,
            mass_kg=260.0,
        )
        pad = crud.create_pad(
            session,
            name="Test Pad",
            stiffness_newtons_per_meter=7.0e7,
            thickness_m=0.01,
        )
        profile = crud.create_support_profile(
            session, name="Test Profile", foundation_modulus_n_per_m2=4.0e7
        )
        project = crud.create_project(session, name="Demo Project", description="Demo")
        config = crud.create_track_config(
            session,
            name="Config A",
            project_id=project.id,
            rail_id=rail.id,
            sleeper_id=sleeper.id,
            pad_id=pad.id,
            support_profile_id=profile.id,
            sleeper_spacing_m=0.6,
            gauge_m=1.435,
        )
        assert config.id is not None

        updated = crud.update_track_config(
            session, config, name="Config B", sleeper_spacing_m=0.65
        )
        assert updated.name == "Config B"
        assert updated.sleeper_spacing_m == pytest.approx(0.65)

        configs = crud.list_track_configs(session)
        assert len(configs) == 1

        crud.delete_track_config(session, updated)
        assert crud.list_track_configs(session) == []


def test_crud_round_trip_for_project() -> None:
    with _make_session() as session:
        project = crud.create_project(
            session,
            name="Demo Project",
            description="Demo",
            vehicle_type="heavy_metro",
            vehicle_subtype="Urban A",
            design_speed_kmh=90.0,
            design_wheel_radius_mm=430.0,
        )
        assert project.id is not None

        updated = crud.update_project(
            session,
            project,
            name="Updated Project",
            description="Updated",
            vehicle_type="high_speed",
            vehicle_subtype=None,
            design_speed_kmh=None,
            design_wheel_radius_mm=450.0,
        )
        assert updated.name == "Updated Project"
        assert updated.description == "Updated"
        assert updated.vehicle_type == "high_speed"
        assert updated.vehicle_subtype is None
        assert updated.design_speed_kmh is None
        assert updated.design_wheel_radius_mm == pytest.approx(450.0)

        projects = crud.list_projects(session)
        assert len(projects) == 1

        crud.delete_project(session, updated)
        assert crud.list_projects(session) == []


def test_crud_round_trip_for_design_alternative_and_project_cascade() -> None:
    with _make_session() as session:
        seed_database(session)
        rail = crud.list_rails(session)[0]
        sleeper = crud.list_sleepers(session)[0]
        pad = crud.list_pads(session)[0]
        support = crud.list_support_profiles(session)[0]
        load_case = crud.list_load_cases(session)[0]
        project = crud.create_project(session, name="Alt Project", description=None)
        config = crud.create_track_config(
            session,
            name="Alt Config",
            project_id=project.id,
            rail_id=rail.id,
            sleeper_id=sleeper.id,
            pad_id=pad.id,
            support_profile_id=support.id,
            sleeper_spacing_m=0.6,
            gauge_m=1.435,
        )

        alternative = crud.create_design_alternative(
            session,
            project_id=project.id,
            track_config_id=config.id,
            load_case_id=load_case.id,
            name="Alt A",
            source_type="sensitivity",
            analysis_type="static",
            changed_parameters={"support_stiffness": 1.2},
            input_snapshot={"track_config": config.name},
            metrics={"max_deflection_m": 0.001},
            status="ok",
            score=0.9,
        )

        assert alternative.id is not None
        updated = crud.update_design_alternative(
            session,
            alternative,
            name="Alt B",
            status="warning",
            metrics={"max_deflection_m": 0.0012},
        )
        assert updated.name == "Alt B"
        assert updated.status == "warning"
        assert len(crud.list_design_alternatives(session, project_id=project.id)) == 1

        crud.delete_project(session, project)
        assert session.query(DesignAlternative).count() == 0


def test_design_alternative_validation_rejects_invalid_status_and_json() -> None:
    with _make_session() as session:
        seed_database(session)
        rail = crud.list_rails(session)[0]
        sleeper = crud.list_sleepers(session)[0]
        pad = crud.list_pads(session)[0]
        support = crud.list_support_profiles(session)[0]
        project = crud.create_project(session, name="Invalid Alt Project", description=None)
        config = crud.create_track_config(
            session,
            name="Invalid Alt Config",
            project_id=project.id,
            rail_id=rail.id,
            sleeper_id=sleeper.id,
            pad_id=pad.id,
            support_profile_id=support.id,
            sleeper_spacing_m=0.6,
            gauge_m=1.435,
        )

        with pytest.raises(ValueError, match="status"):
            crud.create_design_alternative(
                session,
                project_id=project.id,
                track_config_id=config.id,
                name="Bad Status",
                source_type="manual",
                analysis_type="static",
                changed_parameters={},
                input_snapshot={},
                metrics={},
                status="approved",
            )
        with pytest.raises(ValueError, match="JSON serializable"):
            crud.create_design_alternative(
                session,
                project_id=project.id,
                track_config_id=config.id,
                name="Bad JSON",
                source_type="manual",
                analysis_type="static",
                changed_parameters={"bad": {1, 2, 3}},
                input_snapshot={},
                metrics={},
                status="draft",
            )


def test_create_project_rejects_duplicate_name() -> None:
    with _make_session() as session:
        crud.create_project(session, name="Duplicate Project", description=None)
        with pytest.raises(ValueError, match="already exists"):
            crud.create_project(session, name="Duplicate Project", description="Another")


def test_validation_rejects_negative_values() -> None:
    with _make_session() as session:
        with pytest.raises(ValueError, match="elastic_modulus_pa"):
            crud.create_rail(
                session,
                name="Bad Rail",
                elastic_modulus_pa=-1.0,
                moment_inertia_m4=3.05e-5,
                section_modulus_m3=4.1e-4,
                mass_kg_per_m=60.0,
            )

        with pytest.raises(ValueError, match="load_newtons"):
            crud.create_load_case(session, name="Bad Load", load_newtons=-10.0)

        with pytest.raises(ValueError, match="load_newtons"):
            crud.create_load_case(session, name="Zero Load", load_newtons=0.0)


def test_crud_round_trip_for_load_case() -> None:
    with _make_session() as session:
        load_case = crud.create_load_case(
            session,
            name="Load A",
            load_newtons=120_000.0,
            description="Baseline load",
        )
        assert load_case.id is not None

        updated = crud.update_load_case(
            session,
            load_case,
            name="Load B",
            load_newtons=130_000.0,
            description=None,
        )
        assert updated.name == "Load B"
        assert updated.description is None

        load_cases = crud.list_load_cases(session)
        assert len(load_cases) == 1

        crud.delete_load_case(session, updated)
        assert crud.list_load_cases(session) == []


def test_seed_database_inserts_baseline_data() -> None:
    with _make_session() as session:
        seed_database(session)
        rails = crud.list_rails(session)
        sleepers = crud.list_sleepers(session)
        profiles = crud.list_support_profiles(session)
        load_cases = crud.list_load_cases(session)

        assert len(rails) >= 3
        assert len(sleepers) >= 3
        assert len(profiles) >= 3
        assert len(load_cases) >= 2


def test_seed_database_creates_example_projects() -> None:
    with _make_session() as session:
        seed_database(session)
        projects = {project.name: project for project in crud.list_projects(session)}
        for name in ["High Speed Baseline", "Heavy Haul Baseline", "Metro Heavy", "LRT"]:
            assert name in projects
            assert len(projects[name].track_configs) >= 2


def test_seed_database_inserts_rail_profiles_without_overwrite() -> None:
    with _make_session() as session:
        custom = crud.create_rail(
            session,
            name="Custom Rail",
            elastic_modulus_pa=2.1e11,
            moment_inertia_m4=9.9e-5,
            section_modulus_m3=9.9e-4,
            mass_kg_per_m=99.9,
            height_mm=100.0,
            head_width_mm=50.0,
            foot_width_mm=100.0,
            area_cm2=10.0,
        )

        seed_database(session)

        names = {rail.name for rail in crud.list_rails(session)}
        for name in ["AS60", "AS68", "S41", "UIC54", "UIC60", "Ri60"]:
            assert name in names

        refreshed = session.scalar(select(Rail).where(Rail.id == custom.id))
        assert refreshed is not None
        assert refreshed.moment_inertia_m4 == pytest.approx(9.9e-5)
        assert refreshed.section_modulus_m3 == pytest.approx(9.9e-4)
        assert refreshed.mass_kg_per_m == pytest.approx(99.9)
        assert refreshed.height_mm == pytest.approx(100.0)
        assert refreshed.head_width_mm == pytest.approx(50.0)
        assert refreshed.foot_width_mm == pytest.approx(100.0)
        assert refreshed.area_cm2 == pytest.approx(10.0)

        initial_count = len(names)
        seed_database(session)
        assert len(crud.list_rails(session)) == initial_count

        as60 = session.scalar(select(Rail).where(Rail.name == "AS60"))
        assert as60 is not None
        assert as60.moment_inertia_m4 > 0
        assert as60.section_modulus_m3 > 0


def test_seed_database_updates_modern_rail_profiles() -> None:
    expected = {
        "S41": {
            "height_mm": 138.0,
            "head_width_mm": 67.0,
            "foot_width_mm": 125.0,
            "area_cm2": 52.7,
            "mass_kg_per_m": 41.3,
            "iy_m4": 1368.0 * 1.0e-8,
            "iz_m4": 276.0 * 1.0e-8,
            "wyh_m3": 196.0 * 1.0e-6,
            "wyf_m3": 200.5 * 1.0e-6,
            "wz_m3": 44.2 * 1.0e-6,
        },
        "S49": {
            "height_mm": 149.0,
            "head_width_mm": 67.0,
            "foot_width_mm": 125.0,
            "area_cm2": 63.0,
            "mass_kg_per_m": 49.4,
            "iy_m4": 1819.0 * 1.0e-8,
            "iz_m4": 320.0 * 1.0e-8,
            "wyh_m3": 240.0 * 1.0e-6,
            "wyf_m3": 248.0 * 1.0e-6,
            "wz_m3": 51.2 * 1.0e-6,
        },
        "NP46": {
            "height_mm": 142.0,
            "head_width_mm": 72.0,
            "foot_width_mm": 120.0,
            "area_cm2": 59.3,
            "mass_kg_per_m": 46.6,
            "iy_m4": 1605.0 * 1.0e-8,
            "iz_m4": 310.0 * 1.0e-8,
            "wyh_m3": 224.0 * 1.0e-6,
            "wyf_m3": 228.0 * 1.0e-6,
            "wz_m3": 52.0 * 1.0e-6,
        },
        "UIC54": {
            "height_mm": 159.0,
            "head_width_mm": 70.0,
            "foot_width_mm": 140.0,
            "area_cm2": 69.3,
            "mass_kg_per_m": 54.4,
            "iy_m4": 2346.0 * 1.0e-8,
            "iz_m4": 418.0 * 1.0e-8,
            "wyh_m3": 279.0 * 1.0e-6,
            "wyf_m3": 313.0 * 1.0e-6,
            "wz_m3": 60.0 * 1.0e-6,
        },
        "UIC60": {
            "height_mm": 172.0,
            "head_width_mm": 72.0,
            "foot_width_mm": 150.0,
            "area_cm2": 76.9,
            "mass_kg_per_m": 60.3,
            "iy_m4": 3055.0 * 1.0e-8,
            "iz_m4": 513.0 * 1.0e-8,
            "wyh_m3": 336.0 * 1.0e-6,
            "wyf_m3": 377.0 * 1.0e-6,
            "wz_m3": 68.0 * 1.0e-6,
        },
        "Ri60": {
            "height_mm": 180.0,
            "head_width_mm": 113.0,
            "foot_width_mm": 180.0,
            "area_cm2": 77.1,
            "mass_kg_per_m": 60.5,
            "iy_m4": 3334.0 * 1.0e-8,
            "iz_m4": 884.0 * 1.0e-8,
            "wyh_m3": 387.0 * 1.0e-6,
            "wyf_m3": 355.0 * 1.0e-6,
            "wz_m3": 135.0 * 1.0e-6,
        },
    }

    with _make_session() as session:
        seed_database(session)
        for name, values in expected.items():
            rail = session.scalar(select(Rail).where(Rail.name == name))
            assert rail is not None
            assert rail.elastic_modulus_pa == pytest.approx(2.10e11)
            assert rail.height_mm == pytest.approx(values["height_mm"])
            assert rail.head_width_mm == pytest.approx(values["head_width_mm"])
            assert rail.foot_width_mm == pytest.approx(values["foot_width_mm"])
            assert rail.area_cm2 == pytest.approx(values["area_cm2"])
            assert rail.mass_kg_per_m == pytest.approx(values["mass_kg_per_m"])
            assert rail.moment_inertia_m4 == pytest.approx(values["iy_m4"])
            assert rail.moment_inertia_z_m4 == pytest.approx(values["iz_m4"])
            assert rail.section_modulus_head_m3 == pytest.approx(values["wyh_m3"])
            assert rail.section_modulus_foot_m3 == pytest.approx(values["wyf_m3"])
            assert rail.section_modulus_z_m3 == pytest.approx(values["wz_m3"])
            assert rail.section_modulus_m3 == pytest.approx(
                min(values["wyh_m3"], values["wyf_m3"])
            )

        seed_database(session)
        updated = session.scalars(select(Rail).where(Rail.name.in_(expected.keys()))).all()
        assert len(updated) == len(expected)
