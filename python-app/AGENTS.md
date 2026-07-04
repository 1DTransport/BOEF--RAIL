# AGENTS.md

This repo contains:
- A Python desktop application for BOEF track analysis (PySide6 + SQLAlchemy + SQLite + matplotlib).
- Engineering solvers, database migrations/seeds, reproducible examples, and test fixtures.
- A separate `../presentation-workspace/` for presentation tooling when that workspace is directly in scope.

Agents must follow the repo’s conventions, run checks before changes, and keep unit correctness front-and-centre.

---

## 1) Operating principles (non-negotiable)

### Accuracy first
- This project is an engineering tool. Do not “guess” units, equations, or conventions.
- When changing calculations, update tests and include a brief note in the PR describing:
  - which equation changed,
  - what unit convention is used,
  - how you validated it (tests or cross-check).

### Work in small, verifiable steps
- Prefer small PRs with green tests over large refactors.
- If you touch DB schema, include a migration step (or clearly document the schema change + seed update).
- When rail seed data is updated, document whether existing rows are upserted or preserved and how any
  approximations are computed.

### Don't break reproducibility
- Keep sample projects and example outputs stable unless you deliberately update them and note the reason.
- Never commit secrets.
- Documentation sources: always use the OpenAI developer documentation MCP server for OpenAI API/ChatGPT Apps SDK/Codex work, and use Context7 for any other documentation needs.
- Before editing, run `git status --short`. If files are already modified, treat them as user changes and do not overwrite them without understanding the existing diff.

---

## 2) Repo navigation tips

### Primary app layout
- This file lives in `python-app/`, which is the real BOEF app root.
- `app/` - GUI (PySide6).
- `core/` - engineering models, solvers, dynamic/static analysis, exports, and sensitivity logic.
- `db/` - SQLAlchemy models, Alembic migrations, and seed data.
- `tests/` - unit, integration, GUI-adjacent, non-regression, dynamic, transition, and export tests.
- `scripts/build_macos_app.sh` - local PyInstaller macOS app build script.
- `../presentation-workspace/` - JS/TS presentation tooling only; do not apply pnpm/Turbo commands to BOEF app work unless that workspace is directly in scope.
- If structure differs, respect what exists. Don't reorganise unless requested.

---

## 3) Development environment

### BOEF Python app
- Use Python 3.11+ (match `pyproject.toml`).
- Prefer the `./boef` helper because it creates/reuses the virtualenv, installs editable dependencies when `pyproject.toml` changes, sets Qt plugin paths, and runs the app/tests consistently.
- Standard source run from this directory:
  ```bash
  VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef run
  ```
- Standard test command:
  ```bash
  VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef test
  ```
- Standard local macOS app build:
  ```bash
  VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./scripts/build_macos_app.sh
  ```
- Use generic commands such as `python -m app`, `python -m pytest`, or manual virtualenv creation only when there is a concrete reason and the result is verified.
- macOS note: avoid re-running `python -m venv .venv` on top of an existing venv. If Qt plugin errors appear, use the validated `.venv_run` path or recreate the broken venv deliberately.

---

## 4) Unit conventions (critical)

### Internal units
- SI units internally:
  - length: m
  - force: N
  - stress/modulus: Pa
  - moment: N·m

### Unit boundary rule
- Convert once at input parsing.
- Store internal values in SI.
- Convert once at output/export.

### Required checks after modifying engineering code
- Symmetry check (single point load): `y(x) == y(-x)`
- Reaction equilibrium: `∫ p(x) dx ≈ ΣP`
- Derivative coherence:
  - `M(x) = -EI · d²y/dx²`
  - `V(x) = dM/dx`
- Validate against canonical example projects.

---

## 5) Testing instructions

### Default BOEF test command
- Run from `python-app/`:
  ```bash
  VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef test
  ```
- To pass focused pytest selectors, append them after `test`, for example:
  ```bash
  VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef test tests/test_solver.py
  ```
- GUI tests are environment-sensitive. Include them only when explicitly needed and supported:
  ```bash
  BOEF_ENABLE_GUI_TESTS=1 VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef test
  ```
- Lint/format only if configured in the repo; do not introduce a new formatter or linter in an unrelated change.

Quality gate: **do not merge with failing tests**.

---

## 6) Change-type verification matrix

Use the smallest check set that proves the touched behavior, then broaden when the change affects shared contracts or engineering results.

| Change type | Minimum verification | Extra requirements |
| --- | --- | --- |
| Core static solver or engineering equations | `./boef test tests/test_solver.py tests/test_analysis.py tests/test_analysis_engine.py` plus relevant focused tests | State equation impact, SI units, display units, and benchmark/sanity check. Check symmetry, reaction equilibrium, and derivative coherence when applicable. |
| A3902/design checks | `./boef test tests/test_analysis.py tests/test_analysis_regression.py` | Confirm DAF, curve effects, `P_DV`, `Q_R`, ballast/formation pressure logic, and no double counting. |
| Dynamic analysis | `./boef test tests/test_dynamic_solver.py tests/test_dynamic_vs_static_limit.py tests/test_dynamic_transition.py` | Confirm static-limit behavior, damping/critical-speed reasonableness, and dynamic/static isolation. |
| Transition-zone logic or energy outputs | `./boef test tests/test_transition.py tests/test_dynamic_transition.py tests/non_regression/test_non_regression_baselines.py` | Confirm `k(x)` semantics, envelope max-absolute logic, boundary flags, and energy metadata. |
| Stress chart or pressure logic | `./boef test tests/test_stress_metrics.py tests/test_exports.py` | Confirm sign conventions, 2:1 load conservation, compressive-only plot curves, and append-only export columns. |
| Sensitivity logic | `./boef test tests/test_sensitivity.py tests/test_main_window.py -k sensitivity` | Confirm one-variable scenario construction, AS5100 full-train scaling/position shifts, and recommendation text. |
| GUI behavior | Target affected GUI tests, commonly `./boef test tests/test_main_window.py` | Confirm long runs remain in a background worker and new features are optional/toggled where possible. |
| Export/report/schema metadata | `./boef test tests/test_exports.py tests/test_project_io.py` | Preserve existing columns unless deliberately versioned. Add/update schema/version/provenance metadata. |
| Database schema or seed data | `./boef test tests/test_migrations.py tests/test_db_integration.py tests/test_db_crud.py` | Include migration/seed behavior and document whether existing rows are preserved or upserted. |
| macOS packaging | `VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./scripts/build_macos_app.sh` | Verify `dist/BOEF.app`, `Info.plist`, executable type, bundled Alembic resources, and preserve non-app files in `dist/` such as `.icns`. Confirm `CFBundleIconFile=icon-windowed.icns` and compare bundled/source icon hashes when icon behavior is in scope. |
| Documentation-only changes | No app test required unless commands/examples changed | Verify referenced commands, paths, and files exist. Prefer TODOs over invented workflow details. |

When reporting verification, include the exact command run and whether it passed, failed, or was skipped with a reason.

---

## 7) BOEF core engineering requirements

### Static analysis (baseline – must not regress)
- Infinite beam on Winkler foundation (closed-form).
- Superposition for multiple wheel loads.
- Sleeper reactions via tributary integration.
- Damping inputs must be captured/logged but must not change static results.
- AS5100 rail loading must reuse the existing load/envelope machinery:
  - fixed selected arrangement remains available,
  - governing envelope sweep must keep candidate/governing provenance in `run_metadata`,
  - solver load is the per-rail wheel load, i.e. half the axle load for two-rail vertical loading.
- AS5100 sensitivity must preserve the arrangement:
  - wheel-load sensitivity scales every AS5100 axle,
  - AS5100 position sensitivity shifts every axle without changing load magnitudes,
  - governing sweep remains an envelope workflow, not a dynamic candidate sweep.
- A3902 quasi-static design checks should stay available and test-covered when editing design logic:
  - `phi = 1 + delta * eta * t_c`
  - `P_DV = P_SV * phi`
  - `y_max = P_DV * beta / (2k)`
  - `Q_R = F1 * S * k * y_max`
  - `P_A = F2 * Q_R / (B * L_eff)` and optional `P_F` / `P_S` at ballast/fill depths.
  - Track class → VQI mapping: class 1..5 → `[45, 50, 65, 75, 75]`, with `delta = VQI / 200`.
  - Curve effects must not be double counted in A3902: keep `P_SV` as base static wheel load and apply
    curve multipliers exactly once (either via explicit factors or stress path, not both).
  - Standalone pressure helper functions must validate physical inputs (`ballast_depth_m`, `sleeper_width_m`,
    and `effective_bearing_length_m`) before calculation.

### Numerical static solvers
- Live in `python-app/core/solver.py`.
- Include Winkler, Pasternak, discrete supports, two-rail coupling.
- Track modulus estimation in `core/track_modulus.py`.
- Non-uniform k(x) profiles in `core/foundation_profiles.py`.
- Pasternak shear is not supported with Timoshenko beam theory; guard or warn in GUI/solver.

### Transition Zones (Design Metrics)
- Static-only workflow; must not couple into dynamic analysis.
- Non-uniform k(x) transitions require the numerical solver; uniform profiles keep closed form.
- Envelope mode treats load positions as offsets from the reference position `x_ref`.
- Keep chart help content opt-in (Help buttons only; no auto-popups).
- Transition chart annotations should reuse the transition summary metrics on transition render paths
  (deflection/moment/shear/reaction/sleeper/pressure/stress/transition profile) with transparent, non-intrusive
  styling.
- Transition energy outputs are post-processing only; do not change static solver state equations.
- Keep `k(x)` semantics explicit: continuous Winkler modulus in `N/m²` with `q_f(x)=k(x)w(x)` in `N/m`.
- Use naming discipline in code/exports:
  - `u_*` for energy density (`J/m`)
  - `U_*` for integrated energy (`J`)
- Keep the implemented transition-energy equations explicit in docs/exports:
  - `u_rail = M^2/(2EI)`
  - `u_foundation = 0.5*k(x)*w^2`
  - `U = int(u) dx`
- Envelope bending-energy proxies must use the pointwise max-absolute moment envelope before integration.
- Transition profile charts should overlay `u_total` on `k(x)` when Winkler energy results are available.
- Envelope transition energy is an upper-bound proxy unless full time history is available.
- For non-Winkler static foundation models, do not emit Winkler energy outputs; mark outputs as
  model-dependent and keep `energy_metrics`/`energy_series` unset.
- Required transition-energy tests:
  - non-negativity (`u >= 0`, `U >= 0`)
  - max-absolute envelope construction
  - non-uniform derivative and clipped window integration behavior
  - no-regression on legacy transition metrics.

### Stress chart (post-processing only)
- Keep stress calculations outside solver state equations.
- Internal naming and units are mandatory:
  - `sigma_top_fiber_pa`, `sigma_bottom_fiber_pa` (Pa)
  - pressures in Pa, ballast/capping from sleeper-load path.
- Bending sign convention must stay explicit in code/help:
  - positive moment -> top fibre compression positive.
- Pressure sign convention:
  - positive = compression.
- Centralize bearing geometry through one resolver function and export provenance.
- For capping pressure, use 2:1 spread with load conservation:
  - `Q = q_ballast * A_ballast`
  - `q_capping = Q / A_capping`
- Envelope reductions:
  - rail stress uses max-absolute logic.
  - pressure plotting uses compressive-only curves; keep signed/raw values exportable where available.
- Dynamic stress scope:
  - rail bending stress only (peak),
  - sleeper/capping pressure not implemented and must be labeled unavailable.
- Required stress tests:
  - stress sign checks for positive/negative moment
  - 2:1 load-conservation check
  - compressive-only pressure helper never negative
  - no-regression of legacy chart/export leading columns.

### Plot area behavior
- Single view (tabbed) remains the default; All-charts view is optional.
- Do not reparent live plot widgets between views; grid uses cached thumbnails.
- Thumbnails refresh after an analysis run (debounced), not on resize.
- Chart annotation/provenance labels are part of the engineering traceability surface; do not remove them to reduce clutter.
- Label visibility should stay user-controlled through the existing chart label categories:
  - `Inputs`: input/provenance badges and load-position markers.
  - `Outputs`: result/KPI badges and result notes.
  - `Max/min`: plotted-series peak/minimum point labels.
- Dense charts may use below-chart footer labels for traceability/KPI text. Treat footers as presentation only;
  the same `Inputs`/`Outputs` categories should control whether they are populated.
- Dynamic annotation detail remains controlled separately by `Dynamic annotations` (`Full traceability`, `Compact`, `Off`).
- Custom chart axis-family compatibility must remain strict except for static/transition spatial mixing:
  - `STATIC_SPATIAL` and `TRANSITION_SPATIAL` are compatible when x-domains are spatial and interpolable.
  - Keep dynamic time/frequency families isolated from spatial mixes.

---

## 8) Dynamic analysis (NEW – strict isolation rules)

### Purpose
Dynamic analysis models time-dependent rail response under moving loads using **analytical beam-on-elastic-foundation dynamics** (not FEM).

### Non-negotiable isolation
- Dynamic analysis **must not import or depend on**:
  - `core/analysis.py`
  - `core/analysis_engine.py`
- Static and dynamic solvers must remain independent.
- The GUI is the *only* integration layer allowed to call both.

### Required module layout
```
python-app/core/dynamic/
  __init__.py
  config.py       # dynamic inputs (dataclasses)
  results.py      # dynamic outputs + summaries
  solver.py       # analytical dynamic solvers
  engine.py       # backend adapter (GUI entry point)
  validation.py   # dynamic-specific checks
```

### Solver expectations
- Governing equation:
  - Euler–Bernoulli beam on elastic foundation with inertia and damping.
- Moving-load formulation (steady-state in moving frame).
- Frequency-domain / spectral solution is acceptable and preferred.
- Dynamic transition fidelity semantics must remain explicit:
  - `screening` is a uniform-`k1` response approximation.
  - `full_profile` resolves non-uniform `k(x)` directly in the transition solver.
- Dynamic transition `k(x)` outputs/charts must represent the configured transition profile
  (not flattened to uniform `k1`), even when response fidelity is `screening`.
- Outputs must include:
  - w(x, t) – deflection
  - M(x, t) – bending moment
  - V(x, t) – shear force
  - time histories at selected x
  - frequency-domain spectra (FFT / PSD)

### Dynamic validation requirements
- Static limit check: dynamic solution → static BOEF result as v → 0.
- Symmetry check for single moving load.
- Energy / response decay with damping.
- Reasonable amplification near critical speed (no numerical blow-up).

---

## 9) GUI integration rules (dynamic)

- Add a GUI mode selector (e.g. Static / Dynamic).
- GUI branches logic; solvers remain unaware of each other.
- Dynamic results must use **dynamic-specific result objects**.
- Long runs must execute in a background worker.
- Plots must show:
  - axis labels
  - units
  - time or frequency domain clearly indicated.
- Dynamic transition advanced-option guards must enforce supported combinations:
  - Transition mode uses moving-load excitation only.
  - `full_profile` transition requires `ZERO_PAD` boundary mode.
  - `full_profile` transition does not allow irregularity excitation.
  - `full_profile` transition does not allow hysteretic damping.
  - Unsupported combinations should be blocked in UI and validated in backend.

---

## 10) Export and reporting

### Static exports
- CSV: x, y(x), M(x), V(x), p(x) with append-only stress columns.
- Sleeper CSV includes append-only ballast/capping pressure columns (signed + compressive forms).

### Dynamic exports
- Time history CSV:
  - t, w(t), M(t), V(t)
- Frequency domain:
  - f, |W(f)| or PSD
- Reports must state:
  - assumptions
  - governing equations
  - validation checks performed

### Transition exports
- `transition_metrics.csv` (summary + peak locations)
- `transition_series.csv` (x, k(x), w, M, p; envelope max/min if used)
- `transition_run.json` (full inputs + derived values)
- Include schema/version and semantics metadata for transition exports:
  - `transition_metrics_schema_version`
  - `transition_series_schema_version`
  - `k_units`, `k_representation`, `foundation_reaction_law`.
  - boundary-artifact flags and optional normalized-by-`P_ref` fields when available.

### Stress metadata
- Static/envelope export metadata must include:
  - `stress_schema_version`
  - `ballast_thickness_m`
  - `stress_model`
  - `bearing_geometry_provenance`
  - `pressure_sign_convention`

---

## 11) PR instructions

### Title
- `[<project_name>] <concise title>`

### Description checklist
- What changed?
- Tests run?
- Any unit or equation changes?
- Any DB schema changes?

---

## 12) Agent etiquette

- Do not add libraries without justification.
- Prefer standard or already-approved libraries.
- Keep functions small, testable, and typed.
- If unsure about an engineering assumption, **stop and document it** – never guess.
