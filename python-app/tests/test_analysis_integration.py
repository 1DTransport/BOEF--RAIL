from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.analysis import AnalysisInputs, compute_track_response
from core.model import PointLoad
from db import crud
from db.models import Base
from db.seed import seed_database


def test_analysis_pipeline_with_seeded_materials() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        seed_database(session)
        pad = crud.create_pad(
            session,
            name="Baseline Pad",
            stiffness_newtons_per_meter=120_000_000.0,
            thickness_m=0.01,
        )
        rail = crud.list_rails(session)[0]
        sleeper = crud.list_sleepers(session)[0]
        support = crud.list_support_profiles(session)[0]

        assert pad is not None

        inputs = AnalysisInputs(
            loads=[PointLoad(position_m=0.0, load_newtons=120_000.0)],
            foundation_modulus_n_per_m2=support.foundation_modulus_n_per_m2,
            elastic_modulus_pa=rail.elastic_modulus_pa,
            moment_inertia_m4=rail.moment_inertia_m4,
            section_modulus_m3=rail.section_modulus_m3,
            sleeper_spacing_m=0.6,
            sleeper_length_m=sleeper.length_m,
            sleeper_width_m=sleeper.width_m,
            sample_count=151,
        )

        result = compute_track_response(inputs)

    assert len(result.x_m) == inputs.sample_count
    assert any(abs(value) > 0.0 for value in result.deflection_m)
    assert len(result.sleeper_loads_n) == len(result.sleeper_positions_m)
