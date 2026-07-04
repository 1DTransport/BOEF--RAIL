# BOEF Capability Matrix (Before/After) + Verification Evidence

This technical note documents the BOEF capability inventory, gaps, and the
verification evidence for the engineering core in `python-app/core`.

## Scope inventory (current)

| Capability area | Status | Location | Notes |
| --- | --- | --- | --- |
| Foundation models | **Winkler + Pasternak** | `core/model.py`, `core/solver.py` | Closed-form Winkler; Pasternak via numerical solver.
| Beam theory | **Euler–Bernoulli + Timoshenko** | `core/model.py`, `core/solver.py` | Timoshenko available in numerical solver.
| Solver types | **Closed-form + numerical FDM** | `core/model.py`, `core/solver.py` | Numerical grid solver for advanced cases.
| Support modeling | **Continuous + discrete sleepers** | `core/model.py`, `core/solver.py` | Sleeper seat loads via integration; discrete supports in FDM.
| Load modeling | **Point loads + superposition + AS5100** | `core/model.py`, `core/analysis.py`, `core/load_builder.py` | Multi-axle via list of `PointLoad`; AS5100 300LA/150LA fixed and governing sweep modes.
| Parameter estimation | **Implemented** | `core/track_modulus.py` | Track modulus back-calculation utilities.
| Non-uniform k(x) | **Implemented** | `core/foundation_profiles.py` + `core/solver.py` | Numerical solver path.
| Two rails & coupling | **Implemented** | `core/solver.py` | Coupling modeled as linear springs at sleeper nodes.
| Discrete sleepers | **Implemented** | `core/solver.py` | Static discrete support model.
| Dynamics / moving loads | **Implemented** | `core/dynamic/` | Analytical moving-load dynamics.
| Dynamic transition analysis | **Implemented** | `core/dynamic/`, `app/main.py` | Single/envelope modes with screening/full-profile solver fidelity.
| Quasi-static envelopes | **Implemented** | `core/envelope.py` | Max/min response envelopes over moving loads.
| AS5100 governing sweep | **Implemented** | `core/envelope.py`, `core/load_builder.py`, `app/main.py` | Reuses envelope engine, stores candidate/governing provenance, and propagates resolved loads into transition envelope analysis.
| Transition zones (design metrics) | **Implemented** | `core/transition.py`, `app/main.py` | Templates, k(x) profiles, metrics, energy diagnostics, and exports.
| Chart label controls | **Implemented** | `app/main.py` | Inputs, Outputs, and Max/min label categories across static, dynamic, transition, special, and stress charts.
| Chart footer labels | **Implemented** | `app/main.py` | Dense pressure/stress charts can show traceability and KPI summaries below the plot.
| Static depth pressure chart | **Implemented** | `app/main.py`, `core/stress_metrics.py`, `core/analysis.py` | Ballast-top, capping/below-ballast, and A3902 formation/subgrade pressure lines where available.
| Stress chart series controls | **Implemented** | `app/main.py` | Rail stress, ballast pressure, and capping pressure visibility without deleting outputs.
| Sensitivity analysis | **Implemented** | `core/sensitivity.py`, `app/sensitivity_dialog.py`, `app/main.py` | One-variable screening, including AS5100 load scaling and whole-train position shifts.
| Custom multi-axis chart builder | **Implemented** | `app/custom_chart.py`, `app/main.py` | Per-chart custom composition with up to four y-axes.

## Capability gap matrix (required vs. implemented)

| Requirement | Implemented | Location | Risks / assumptions |
| --- | --- | --- | --- |
| A) Winkler BOEF closed form | **Implemented** | `core/model.py` | Behavior is contractual; changes must preserve existing outputs.
| B) Track modulus estimation | **Implemented** | `core/track_modulus.py` | Requires deflection measurements in SI; root-bracket must contain solution.
| C) Pasternak foundation | **Implemented** | `core/solver.py` | Numerical FDM; kg→0 must reduce to Winkler.
| D) Discrete supports | **Implemented** | `core/solver.py` | Sleeper positions must align with grid spacing; static only.
| E) Two rails + coupling | **Implemented** | `core/solver.py` | Coupling modeled as linear springs at sleeper nodes.
| F) Non-uniform k(x) | **Implemented** | `core/foundation_profiles.py` + `core/solver.py` | Requires numerical solver; no closed form.
| G) Dynamics + moving loads | **Implemented** | `core/dynamic/` | Isolated dynamic solver; GUI selects static vs dynamic.
| G1) Dynamic transition mode | **Implemented** | `core/dynamic/`, `app/main.py` | Dedicated dynamic transition workflow and exports.
| H) Quasi-static envelopes | **Implemented** | `core/envelope.py` | Envelope max/min and summary outputs.
| H1) AS5100 governing envelope sweep | **Implemented** | `core/envelope.py`, `core/load_builder.py`, `app/main.py` | Existing envelope path is reused; metadata records selected and governing arrangement details.
| I) Transition zone metrics | **Implemented** | `core/transition.py`, `app/main.py` | Template-driven design metrics, Winkler energy diagnostics, dual-axis `k(x)`/`u_total` chart, and exports.
| I1) Transition envelope with AS5100 sweep | **Implemented** | `app/main.py`, `core/envelope.py` | Transition envelope receives `as5100_sweep`; resolved governing loads and metadata are used for transition metrics and labels.
| J) Global custom chart builder | **Implemented** | `app/custom_chart.py`, `app/main.py` | Same-domain multi-axis chart composition.
| K) Chart-label visibility | **Implemented** | `app/main.py` | User can show/hide input/provenance, output/KPI, and max/min labels independently.
| L) AS5100 sensitivity screening | **Implemented** | `core/sensitivity.py`, `app/sensitivity_dialog.py` | Wheel-load sensitivity scales every AS5100 axle; AS5100 position sensitivity shifts the full selected fixed arrangement.
| M) Continuous modulus display units | **Implemented** | `core/units.py`, Alembic migration `0010`, `app/main.py` | UI/export labels use MN/m² while core state remains N/m².

## Unit convention audit + enforcement plan

- **Internal units**: SI (m, N, Pa, s) per `core/units.py` and `python-app/README.md`.
- **Boundary conversion**: UI input/output converts once at the boundary (see `app/main.py`).
- **Plan**:
  1. Keep all new solver and estimation utilities in SI units only.
  2. Validate input arrays in SI; enforce non-negativity where physically required.
  3. Add tests that assert known SI outputs (see Verification Evidence).

## Refactor risk list

- **Closed-form regression risk**: `core/model.py` outputs are used by the GUI and tests.
  - Mitigation: preserve equations; add regression tests against existing outputs.
- **Unit handling risk**: mixing kN/mm or MPa with SI could silently break results.
  - Mitigation: all new APIs are SI-only; tests cover SI assumptions.
- **Solver boundary effects**: finite domain truncation can bias numeric results.
  - Mitigation: default domains use `10 / beta` (same as existing analysis range) and tests allow tolerance.
- **Performance risk**: dense solver can be slow on very large grids.
  - Mitigation: keep grid sizes modest; future improvement can add banded solver.

## Verification evidence (tests)

- **T0 Backwards compatibility**:
  - Existing `tests/test_core.py` regression tests retained.
- **T1 Closed-form validation (Winkler)**:
  - `tests/test_solver.py::test_fdm_matches_closed_form_winkler` compares FDM to closed-form.
- **T2 Cai et al. trend reproduction (proxy)**:
  - `tests/test_track_modulus.py` verifies modulus estimation recovers known k.
- **T3 Pasternak reduction**:
  - `tests/test_solver.py::test_pasternak_reduces_to_winkler_when_shear_zero`.
- **T4 Discrete vs equivalent continuous**:
  - `tests/test_solver.py::test_discrete_supports_close_to_equivalent_continuous`.
- **T5 Unit tests**:
  - `tests/test_core.py::test_unit_conversions_round_trip`.
- **T6 Dynamic solver checks**:
  - `tests/test_dynamic_solver.py` (static limit, symmetry, equilibrium, damping).
- **T7 Transition metrics checks**:
  - `tests/test_transition.py` (profile monotonicity, metric correctness, energy metadata, envelope energy max-absolute logic).
- **T8 Dynamic transition checks**:
  - `tests/test_dynamic_transition.py` (single/envelope behavior, static limit, full-profile parity).
- **T9 Custom chart checks**:
  - `tests/test_custom_chart.py` (axis-family validation and interpolation).
- **T10 AS5100 sweep and provenance checks**:
  - `tests/test_envelope.py` and `tests/test_main_window.py` cover governing sweep reuse, AS5100 metadata, fixed-arrangement preservation, and transition-envelope metadata handoff.
- **T11 Chart-label visibility checks**:
  - `tests/test_main_window.py` covers input/output overlay filtering, load-marker hiding, dynamic output-only annotations, and max/min label rerendering.
- **T12 Sensitivity checks**:
  - `tests/test_sensitivity.py` and `tests/test_main_window.py` cover AS5100 wheel-load scaling, AS5100 position shifts, and sensitivity dialog labelling.
- **T13 Pressure and unit-label checks**:
  - `tests/test_main_window.py` covers static pressure depth series and below-chart footer labels.
  - `tests/test_migrations.py` covers support-profile default names being migrated from `MN/m` to `MN/m²`.

## Notes / follow-ups

- Transition energy diagnostics are static Winkler post-processing only. The implemented equations are
  `u_rail = M^2/(2EI)`, `u_foundation = 0.5*k(x)*w^2`, and `U = int(u) dx`; non-Winkler transition runs leave
  energy outputs blank and labelled model-dependent.
- Transition metrics exports now include energy theory metadata and the transition profile chart overlays
  `u_total` on `k(x)` when those results are available.
- AS5100 governing sweep is a wrapper/provenance layer over the existing envelope engine, not a duplicate solver.
- AS5100 fixed arrangements can be used as ordinary load sets in static/dynamic workflows, while governing sweep
  remains an envelope-specific workflow.
- Chart-label controls are presentation controls only; they hide rendered labels by category and do not change
  solver outputs, exports, or stored metadata.
- Footer labels are also presentation controls. They move dense chart metadata/KPI text below selected plots but
  do not change calculations or exports.
- Sensitivity results are screening outputs. They help identify influential parameters or reference-position
  sensitivity, but they are not optimisation or standards-compliance evidence by themselves.
- Transition metrics and envelope runs are compute-heavy; consider a banded solver or caching if grid sizes grow.
