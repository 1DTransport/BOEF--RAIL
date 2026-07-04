import os
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

if os.environ.get("BOEF_ENABLE_GUI_TESTS", "").lower() not in {"1", "true", "yes"}:
    pytest.skip("Set BOEF_ENABLE_GUI_TESTS=1 to run PySide GUI tests.", allow_module_level=True)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip(
    "PySide6.QtWidgets",
    reason="PySide6 runtime dependencies are missing",
    exc_type=ImportError,
)
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from app.main import LoadCaseDialog, MaterialDialog, MaterialField, TrackConfigDialog  # noqa: E402
from db import crud  # noqa: E402
from db.models import Base  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _qt_offscreen() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def test_material_dialog_disables_delete_without_selection(qapp: QApplication) -> None:
    class _Item:
        def __init__(self) -> None:
            self.name = "Sample"
            self.elastic_modulus_pa = 123.0

    dialog = MaterialDialog(
        MagicMock(),
        title="Test",
        list_items=lambda _session: [_Item()],
        create_item=lambda *_args, **_kwargs: None,
        update_item=lambda *_args, **_kwargs: None,
        delete_item=lambda *_args, **_kwargs: None,
        fields=[
            MaterialField(
                key="elastic_modulus_pa",
                label="Elastic modulus",
                unit="Pa",
                to_si=lambda value: value,
                from_si=lambda value: value,
            )
        ],
    )

    assert not dialog.delete_button.isEnabled()
    dialog.list_widget.setCurrentRow(0)
    assert dialog.delete_button.isEnabled()
    dialog._clear_selection()
    assert not dialog.delete_button.isEnabled()


def test_load_case_dialog_disables_delete_without_selection(qapp: QApplication) -> None:
    with _make_session() as session:
        crud.create_load_case(session, name="Load A", load_newtons=1_000.0, description=None)
        dialog = LoadCaseDialog(session)
        assert not dialog.delete_button.isEnabled()
        dialog.list_widget.setCurrentRow(0)
        assert dialog.delete_button.isEnabled()


def test_material_dialog_inline_validation_marks_invalid(qapp: QApplication) -> None:
    create_item = MagicMock()
    dialog = MaterialDialog(
        MagicMock(),
        title="Test",
        list_items=lambda _session: [],
        create_item=create_item,
        update_item=lambda *_args, **_kwargs: None,
        delete_item=lambda *_args, **_kwargs: None,
        fields=[
            MaterialField(
                key="elastic_modulus_pa",
                label="Elastic modulus",
                unit="Pa",
                to_si=lambda value: value,
                from_si=lambda value: value,
            )
        ],
    )

    dialog.name_input.setText("Temp")
    dialog.name_input.setText("")
    dialog.field_inputs["elastic_modulus_pa"].set_value(1.0)
    dialog.field_inputs["elastic_modulus_pa"].set_value(0.0)

    assert dialog.validation_labels["name"].isVisible()
    assert "required" in dialog.validation_labels["name"].text().lower()
    assert dialog.validation_labels["elastic_modulus_pa"].isVisible()
    assert "> 0" in dialog.validation_labels["elastic_modulus_pa"].text()

    dialog._save()
    assert create_item.call_count == 0

def test_material_dialog_rolls_back_on_commit_failure(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = MagicMock()

    def create_item(_session: MagicMock, **_values: float) -> None:
        raise SQLAlchemyError("db is down")

    dialog = MaterialDialog(
        session,
        title="Test",
        list_items=lambda _session: [],
        create_item=create_item,
        update_item=lambda *_args, **_kwargs: None,
        delete_item=lambda *_args, **_kwargs: None,
        fields=[
            MaterialField(
                key="elastic_modulus_pa",
                label="Elastic modulus",
                unit="Pa",
                to_si=lambda value: value,
                from_si=lambda value: value,
            )
        ],
    )

    dialog.name_input.setText("Test Rail")
    dialog.field_inputs["elastic_modulus_pa"].set_value(1.0)

    critical_mock = MagicMock()
    monkeypatch.setattr(QMessageBox, "critical", critical_mock)

    dialog._save()

    session.rollback.assert_called_once()
    critical_mock.assert_called_once()
    _, _, message = critical_mock.call_args.args
    assert "Unable to save changes" in message
    assert "db is down" not in message
    assert "SQLAlchemyError" not in message
    assert dialog.list_widget.count() == 0
    assert dialog.name_input.text() == "Test Rail"


def test_load_case_dialog_create_and_delete(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _make_session() as session:
        dialog = LoadCaseDialog(session)
        dialog.name_input.setText("Test Load")
        dialog.load_input.set_value(125.0)
        dialog.description_input.setText("Demo case")

        dialog._save()

        assert dialog.list_widget.count() == 1
        assert crud.list_load_cases(session)[0].name == "Test Load"

        dialog.list_widget.setCurrentRow(0)

        monkeypatch.setattr(
            QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes
        )
        dialog._delete()

        assert dialog.list_widget.count() == 0
        assert crud.list_load_cases(session) == []


def test_track_config_dialog_validation_error(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _make_session() as session:
        rail = crud.create_rail(
            session,
            name="Rail",
            elastic_modulus_pa=2.1e11,
            moment_inertia_m4=3.05e-5,
            section_modulus_m3=4.1e-4,
            mass_kg_per_m=60.0,
        )
        sleeper = crud.create_sleeper(
            session,
            name="Sleeper",
            elastic_modulus_pa=3.2e10,
            length_m=2.5,
            width_m=0.25,
            height_m=0.21,
            mass_kg=260.0,
        )
        pad = crud.create_pad(
            session,
            name="Pad",
            stiffness_newtons_per_meter=7.0e7,
            thickness_m=0.01,
        )
        profile = crud.create_support_profile(
            session, name="Profile", foundation_modulus_n_per_m2=4.0e7
        )
        project = crud.create_project(session, name="Project", description=None)

        dialog = TrackConfigDialog(session)
        dialog.name_input.setText("Config")
        dialog.project_combo.setCurrentIndex(0)
        dialog.rail_combo.setCurrentIndex(0)
        dialog.sleeper_combo.setCurrentIndex(0)
        dialog.pad_combo.setCurrentIndex(0)
        dialog.support_combo.setCurrentIndex(0)
        dialog.sleeper_spacing_input.set_value(600.0)
        dialog.gauge_input.set_value(1435.0)

        monkeypatch.setattr(dialog.gauge_input, "value", lambda: -1.0)

        dialog._save()

        assert dialog.validation_labels["gauge"].isVisible()
        assert crud.list_track_configs(session) == []


def test_load_case_dialog_inline_validation_marks_invalid(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _make_session() as session:
        dialog = LoadCaseDialog(session)
        dialog.name_input.setText("Temp")
        dialog.name_input.setText("")
        monkeypatch.setattr(dialog.load_input, "value", lambda: 0.0)

        dialog._validate_fields(force=True)

        assert dialog.validation_labels["name"].isVisible()
        assert dialog.validation_labels["load"].isVisible()


def test_load_case_dialog_save_blocks_invalid_input(qapp: QApplication) -> None:
    with _make_session() as session:
        dialog = LoadCaseDialog(session)
        dialog.name_input.setText("")
        dialog.load_input.set_value(0.0)

        dialog._save()

        assert crud.list_load_cases(session) == []
