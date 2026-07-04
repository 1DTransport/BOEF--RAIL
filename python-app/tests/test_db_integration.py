from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.models import Base, Rail


def test_create_tables_and_insert_rail() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        rail = Rail(
            name="Test Rail",
            elastic_modulus_pa=2.1e11,
            moment_inertia_m4=3.1e-5,
            section_modulus_m3=4.0e-4,
            mass_kg_per_m=60.0,
        )
        session.add(rail)
        session.commit()

        loaded = session.get(Rail, rail.id)
        assert loaded is not None
        assert loaded.name == "Test Rail"
