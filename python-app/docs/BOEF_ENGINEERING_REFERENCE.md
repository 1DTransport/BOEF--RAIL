# BOEF Engineering Reference

## Introduction

The BOEF application is a desktop engineering tool for railway track analysis where rails are modeled as beams and support layers are represented with continuous or discrete elastic foundations. It combines closed-form and numerical methods so engineers can evaluate deflection, moment, shear, support reactions, sleeper demand, and stress/pressure outputs in a single workflow.

The codebase is organized to keep physics paths explicit and auditable: static solvers, dynamic solvers, transition metrics, stress post-processing, GUI orchestration, and reproducible exports are separated into clear modules. This allows day-to-day design checks and deeper investigations without requiring full 3D finite-element models for routine tasks.

BOEF is intended to support engineering delivery and review with transparent equations, SI-unit consistency, standards-aligned checks (including A3902-aligned paths), and export artifacts suitable for traceable reporting.

## Codebase Summary Snapshot

- `app/`: PySide6 GUI, worker orchestration, chart rendering, mode routing (Static, Dynamic, Special).
- `core/`: engineering calculations and engines.
- `core/dynamic/`: dynamic-only solver stack (isolated from static engine modules).
- `core/load_builder.py`: train/axle and AS5100 rail load construction plus load-source provenance.
- `db/`: SQLAlchemy models, CRUD, migrations, seed data, project I/O.
- `tests/`: solver, GUI, export, transition, dynamic, and non-regression coverage.
- `boef`: local runner for `run`, `core`, `db`, and `test` workflows.

### Core entry points

- GUI app: `python -m app`
- Engineering CLI: `python -m core`
  - quick deflection: `python -m core --load-newtons <N> --stiffness-newtons-per-meter <N/m>`
  - structured subcommands: `python -m core deflection ...` and `python -m core a3902 ...`
- Database CLI: `python -m db --sqlite-path <path>`

## Purpose and Audience

This document is a technical reference for engineers using the BOEF desktop application for railway track analysis.
It is intended for:

- track and rail engineers performing design checks,
- reviewers auditing assumptions and equation paths,
- teams preparing technical presentations or handover packages.

This is a theory and capability document. It does not control runtime behavior.

## 1. Product Scope

BOEF combines analytical and numerical railway track analysis in one desktop workflow. The application supports:

- Static infinite-beam-on-elastic-foundation (BOEF) analysis.
- Quasi-static moving-load envelope analysis.
- AS5100 300LA/150LA rail loading with fixed selected and governing envelope sweep modes.
- Static transition-zone design metrics with non-uniform support `k(x)`.
- Numerical static extensions:
  - Pasternak shear layer,
  - discrete supports,
  - two-rail coupling,
  - Timoshenko beam option,
  - multilayer equivalent support models.
- Dynamic moving-load analysis (steady-state/time-history with FFT/PSD outputs).
- Dynamic transition analysis (screening and full-profile fidelity).
- Dipped-joint wheel/rail force analysis.
- Floating-slab isolation analysis.
- Sensitivity analysis for one-variable-at-a-time screening, including AS5100 load scale and train-position
  scenarios.
- Stress and pressure post-processing for rail and support layers.
- Chart label visibility controls for input/provenance, output/KPI, and max/min point labels.
- Below-chart footer labels for dense pressure/stress plots where overlay labels would obscure the engineering
  data.
- Export and reproducibility metadata for engineering audit trails.

## 2. Architecture and Separation of Concerns

## 2.1 High-level architecture

- GUI layer: `app/main.py`
  - collects inputs, validates UI state, dispatches workers, renders charts/tables.
- Static engine: `core/analysis_engine.py`
  - selects closed-form vs numerical static backend.
- Dynamic engine: `core/dynamic/engine.py`
  - dispatches moving-load, transition, and dipped-joint dynamic workflows.
- Special engine: `core/special/engine.py`
  - dispatches floating-slab workflow.
- Data layer: `db/models.py`, `db/crud.py`, `db/project_io.py`, migrations and seed scripts.

## 2.2 Static vs dynamic isolation

Dynamic analysis is intentionally isolated from static solver implementation logic:

- dynamic modules live in `core/dynamic/`,
- static workflows live in `core/analysis.py`, `core/analysis_engine.py`, `core/solver.py`,
- GUI is the integration layer that can call both.

This prevents equation mixing and keeps validation paths mode-specific.

## 2.3 Main engineering modules

- `core/model.py`: closed-form Winkler BOEF equations and superposition.
- `core/load_builder.py`: train/axle and AS5100 rail load builders with traceability metadata.
- `core/analysis.py`: static analysis pipeline, design checks (including A3902-aligned values).
- `core/solver.py`: finite-difference static solvers (Euler/Timoshenko, Pasternak, discrete, two-rail).
- `core/envelope.py`: quasi-static envelope sweeps.
- `core/transition.py`: transition metrics, energy metrics, and series assembly.
- `core/stress_metrics.py`: rail stress + ballast/capping pressure post-processing.
- `core/dynamic/solver.py`: travelling-wave moving-load solver, time-series/frequency-output builder, dipped-joint equations.
- `core/dynamic/validation.py`: dynamic input and mode compatibility checks.
- `core/special/solver.py`: floating-slab SDOF isolation solver.
- `core/track_modulus.py`: track modulus estimation/back-calculation methods.
- `core/sensitivity.py`: one-variable scenario generation, scoring, and recommendation text.

## 3. Unit System and Sign Conventions

## 3.1 Internal unit convention (SI)

- Length: `m`
- Force: `N`
- Stress and modulus: `Pa`
- Moment: `N·m`
- Mass: `kg`, `kg/m`
- Time: `s`

Input conversions are intended at boundaries (UI/import). Internal computations are SI.

## 3.2 Stress and pressure conventions

From `core/stress_metrics.py`:

- Rail top fiber:
  - `sigma_top_fiber_pa = M / Z_top`
- Rail bottom fiber:
  - `sigma_bottom_fiber_pa = -M / Z_bottom`
- Positive pressure means compression.
- Capping pressure uses 2:1 spread with load conservation.
- Continuous foundation modulus is stored internally in `N/m^2` and displayed/exported as `MN/m^2`.

## 4. Static BOEF Theory (Closed Form)

Primary implementation: `core/model.py`.

For an Euler-Bernoulli beam on Winkler foundation:

\[
\beta = \left(\frac{k}{4EI}\right)^{1/4}
\]

where:

- `k`: foundation modulus (`N/m^2`),
- `E`: rail elastic modulus (`Pa`),
- `I`: rail second moment of area (`m^4`).

Characteristic distances:

\[
x_1 = \frac{\pi}{4\beta}, \quad x_{cf} = 3x_1
\]

Single-load extrema:

\[
M_{max} = \frac{P}{4\beta}, \quad y_{max} = \frac{P\beta}{2k}
\]

Point-load responses are implemented with exponential-decay harmonic terms; multiple loads are handled by linear superposition:

\[
w(x) = \sum_i w_i(x), \quad M(x)=\sum_i M_i(x), \quad V(x)=\sum_i V_i(x), \quad q(x)=k\,w(x)
\]

Sleeper seat loads are computed by integrating reaction over tributary lengths.

## 5. Static Numerical Solver Theory and Extensions

Primary implementation: `core/solver.py`, orchestrated by `core/analysis_engine.py`.

## 5.1 Euler-Bernoulli finite-difference solver

Governing equation:

\[
EI\,w''''(x) - k_g\,w''(x) + k_s\,w(x) = q(x)
\]

- `k_g = 0` gives Winkler form.
- `k_s` can be uniform or spatially varying profile.
- boundary treatment enforces decay-compatible constraints in finite domain.

## 5.2 Timoshenko finite-difference solver

Unknowns: deflection `w(x)` and rotation `phi(x)`.

\[
EI\,\phi'' = \kappa G A (\phi - w')
\]
\[
\frac{d}{dx}\left[\kappa G A(\phi - w')\right] + k_s w = q
\]

Implementation solves the Timoshenko branch directly. Results include:

- `slope_rad = w'`
- `rotation_rad = phi`
- `shear_angle_rad = phi - w'`

This distinction matters because Euler-Bernoulli assumes rail slope and cross-section rotation are the same,
whereas Timoshenko treats them as independent fields.

## 5.3 Pasternak support

Pasternak shear layer contribution is included in static numerical formulations through `k_g` terms.

The reported total support reaction is:

\[
r(x)=k_s w(x)-k_g w''(x)
\]

BOEF keeps the Winkler component and Pasternak shear-layer component separately available in numerical results
as `winkler_reaction_n_per_m` and `pasternak_shear_reaction_n_per_m`. Beam shear `V` remains the internal
rail shear derived from bending moment; it is not the same quantity as the Pasternak foundation shear-layer
reaction component.

Constraint:

- Pasternak + Timoshenko is blocked in this release (`core/analysis_engine.py`).

## 5.4 Discrete supports

Discrete springs can be mapped to grid nodes (sleepers/pads), enabling comparison against equivalent continuous support assumptions.

## 5.5 Two-rail coupled static solver

Left/right rails are solved in one coupled system with cross-rail spring coupling at specified nodes (usually sleeper locations).

## 5.6 Multilayer static equivalent support models

From `core/foundation/base.py` and `core/analysis_engine.py`:

Series stiffness:

\[
k_{eq} = \frac{k_{pad}\,k_{bed}}{k_{pad}+k_{bed}}
\]

Per-support values are converted to per-length using sleeper spacing.

Modes:

- `series` (railpad + trackbed),
- `sleeper_mass` (static equivalent reporting path).

## 6. Static Design Checks (A3902-Aligned Path)

Primary implementation: `core/analysis.py`.

Core factors:

\[
\phi = 1 + \delta \eta t_c
\]
\[
P_{DV} = P_{SV}\,\phi
\]
\[
y_{max} = \frac{P_{DV}\,\beta}{2k}
\]
\[
Q_R = F_1 S k y_{max}
\]
\[
P_A = \frac{F_2 Q_R}{B L_{eff}}
\]

Formation/subgrade follow depth-spread expressions in `formation_pressure_a3902` and `subgrade_pressure_a3902`.

Track-class to VQI mapping in code:

- class `1..5` maps to `[45, 50, 65, 75, 75]`,
- `delta = VQI / 200`.

Additional checks include bending/shear admissibility and combined stress envelopes.

## 7. Quasi-Static Envelope Analysis

Primary implementation: `core/envelope.py`.

Envelope mode shifts load groups over `x_ref` positions and tracks max/min series for:

- deflection,
- moment,
- shear,
- reaction,
- sleeper loads,
- ballast pressure,
- formation/subgrade pressure by depth.

Output includes per-series extrema and summary metrics with location tracking.

Supports closed-form and numerical backend envelope pathways (with compatibility constraints).

## 7.1 AS5100 rail loading and governing sweep

Primary implementations: `core/load_builder.py`, `core/envelope.py`, and `app/main.py`.

AS5100 loading is implemented as a load-source and envelope workflow rather than a separate solver:

- `Fixed selected arrangement` builds the selected 300LA/150LA axle group layout.
- `Governing envelope sweep` runs compact candidate arrangements through the existing `run_envelope(...)`
  engine and selects the governing response by the recorded governing metric.
- The solver load remains the per-rail wheel load. For axle loading this is half the axle load, and the UI
  load markers expose both the wheel load per rail and the parent axle load.
- `run_metadata` records the standard, model, selected arrangement, governing arrangement, candidate count,
  reference position `x0`, and governing metric.

Manual single-load and multiple-load inputs are interpreted as per-rail wheel loads already acting on one rail.
Train/axle-builder and AS5100 workflows are interpreted as total axle loads and then split to the per-rail wheel
line before the BOEF solver is called.

When AS5100 governing sweep is used with static transition-envelope analysis, the resolved governing loads are
propagated into the transition context before transition metrics are assembled. This keeps transition summaries,
chart labels, stress post-processing, and exports aligned to the actual governing wheel arrangement.

AS5100 fixed arrangements can also be used as ordinary load sets in static, dynamic, dynamic transition, and
sensitivity workflows. Governing sweep remains an envelope-specific workflow; dynamic analysis uses the selected
fixed AS5100 arrangement and does not perform an AS5100 candidate sweep.

## 8. Transition-Zone Static Analysis

Primary implementation: `core/transition.py`.

## 8.1 `k(x)` profile types

- uniform,
- step,
- ramp,
- exponential,
- segment.

## 8.2 Transition metrics

Computed metrics include:

- `Delta w(s)` and `Delta w(1m)`,
- max curvature,
- max moment,
- bending energy proxy,
- max reaction gradient,
- max sleeper load.

## 8.3 Transition energy outputs (Winkler-consistent post-processing)

Energy density definitions:

\[
u_{rail}(x)=\frac{M(x)^2}{2EI}
\]
\[
u_{foundation}(x)=\frac{1}{2}k(x)w(x)^2
\]
\[
u_{total}=u_{rail}+u_{foundation}
\]

Integrated energies:

\[
U_{rail}=\int u_{rail}\,dx,\quad U_{foundation}=\int u_{foundation}\,dx,\quad U_{total}=U_{rail}+U_{foundation}
\]

These outputs are a line-beam Winkler surrogate for energy concentration in a stiffness transition. They are
consistent with Euler-Bernoulli bending strain energy and linear Winkler spring energy, but they are not a
replacement for full 2D/3D layer-by-layer finite-element strain-energy assessment. Use them to identify where a
static transition response concentrates rail/foundation elastic energy and where the concentration changes rapidly.

Additional diagnostics include:

- energy partition `eta = U_foundation / U_total`,
- max energy density location,
- max `|du/dx|`,
- windowed energy/average over target window length.

Envelope energy is explicitly marked as an upper-bound proxy in result metadata. The bending-energy proxy uses
the pointwise max-absolute moment envelope before integration, so opposite-sign governing moments at different
locations remain represented in the reported energy.

Transition exports include the energy-method metadata fields `energy_method`, `energy_equations`, and
`energy_scope`, and the transition profile chart overlays `u_total` on `k(x)` when Winkler energy results are
available.

## 9. Stress and Pressure Post-Processing

Primary implementation: `core/stress_metrics.py`.

Stress outputs are intentionally post-processing only and do not modify solver state equations.

## 9.1 Rail bending stress

\[
\sigma_{top}=\frac{M}{Z_{top}}, \quad \sigma_{bottom}=-\frac{M}{Z_{bottom}}
\]

## 9.2 Ballast pressure

\[
q_{ballast} = \frac{Q}{A_{bearing}}
\]

## 9.3 Capping pressure (2:1 load-conserving spread)

\[
Q = q_{ballast}A_{ballast}
\]
\[
q_{capping} = \frac{Q}{(B+2h_b)(L+2h_b)}
\]

where `h_b` is ballast thickness.

Static pressure charts can show several pressure levels together:

- ballast top / sleeper contact pressure,
- below-ballast or capping-top pressure from the load-conserving 2:1 spread,
- A3902 formation pressure where the ballast-depth design input exists,
- A3902 subgrade pressure where the fill-depth/design inputs exist and the value differs from formation pressure.

## 9.4 Envelope reductions

- Rail stress envelope uses max-absolute logic.
- Pressure envelope plotting uses compressive-only curves.
- Signed pressure values remain available for export where emitted.

## 9.5 Dynamic stress scope

Dynamic mode reports rail bending stress (peak) only.
Sleeper/ballast/capping pressure in dynamic mode is intentionally unavailable in this release.

## 10. Dynamic Moving-Load Analysis

Primary implementation: `core/dynamic/solver.py`, `core/dynamic/engine.py`, `core/dynamic/validation.py`.

## 10.1 Governing formulation

In the travelling-wave moving-frame solution, the solver assembles the following wavenumber-domain denominator internally and then returns spatial response fields in \(\xi\):

\[
EI k^4 - m v^2 k^2 + i c v k + k_f + G k^2
\]

with:

- `m`: rail mass per length,
- `v`: speed,
- `c`: viscous foundation damping,
- `k_f`: foundation modulus,
- `G`: Pasternak shear term.

For hysteretic damping mode, complex stiffness is used:

\[
k_f(1+i\eta)
\]

where `eta` is loss factor.

## 10.2 Load and excitation options

- Moving-load excitation (default).
- Moving-oscillator transfer modifier (bounded transfer factor).
- Optional irregularity forcing:
  - profile interpolation mode,
  - synthetic PSD-inspired profile mode (`~ C/k^2` in implementation).

## 10.3 Spatial outputs

- `w(xi)` deflection,
- `M(xi)` moment,
- `V(xi)` shear,
- `q(xi)` reaction,
- damping-force series.

## 10.4 Probe time series and spectra

At selected probe positions:

- time histories for `w, M, V, q`,
- FFT amplitude spectrum,
- Welch PSD with confidence intervals,
- support impedance magnitude and phase.

## 10.5 Dynamic risk and speed metrics

Engine computes critical speed estimate:

\[
v_{cr}=\sqrt{\frac{2\sqrt{EI\,k}}{m}}
\]

and reports:

- `critical_speed_ratio = v / v_cr`,
- dynamic amplification vs static deflection baseline,
- risk index and risk map (when transition stiffness ratio is provided).

## 11. Dynamic Transition Analysis

Primary implementation: `core/dynamic/engine.py` + `core/dynamic/solver.py`.

## 11.1 Run modes

- single,
- envelope over `x_ref` range.

## 11.2 Fidelity modes

- `screening`:
  - uniform `k1` response path for demand screening.
- `full_profile`:
  - non-uniform transition profile solved directly in moving-frame finite-difference system.

## 11.3 Full-profile compatibility constraints

Validated in `core/dynamic/validation.py`:

- no periodic boundary mode,
- no irregularity excitation,
- no hysteretic damping,
- no moving-oscillator mode.

## 11.4 Outputs

- dynamic transition metrics:
  - max deflection/moment/shear/reaction,
  - governing `x_ref`,
  - risk index and critical speed ratio.
- transition series:
  - `k(x)` plus single or envelope response arrays.

## 12. Dynamic Dipped-Joint Analysis

Primary implementation: `solve_dipped_joint_forces` in `core/dynamic/solver.py`.

This mode computes wheel/rail peak forces using Jenkins/Cope-style relations with:

- static wheel load,
- dip angle input (`2 alpha`),
- speed,
- Hertzian stiffness,
- unsprung/track masses,
- equivalent track stiffness and damping.

Outputs:

- `P1`, `P2`,
- dynamic amplification factors at both points.

## 13. Special Mode: Floating Slab Isolation

Primary implementation: `core/special/solver.py`.

Floating slab is modeled as SDOF isolation system:

- natural frequency:
\[
f_n = \frac{1}{2\pi}\sqrt{\frac{k_{eff}}{m}}
\]
- damping ratio:
\[
\zeta = \frac{c}{2\sqrt{k_{eff}m}}
\]

If railpad stiffness is supplied, effective stiffness uses series combination with isolator stiffness.

Frequency sweep outputs:

- transmissibility,
- attenuation in dB,
- static deflection.

## 14. Track Modulus Estimation Utilities

Primary implementation: `core/track_modulus.py`.

Supported methods include:

- Deflection-area method:
\[
k = \frac{P}{\int w(x)\,dx}
\]
- Two-load delta-area method:
\[
k = \frac{P_1-P_2}{\int (w_1-w_2)\,dx}
\]
- Single-point inversion (root solve) based on closed-form `w(0)` expression.
- Synthesis from spring constants and sleeper spacing.

These functions support calibration, verification, and back-analysis workflows.

## 15. Sensitivity Analysis

Primary implementation: `core/sensitivity.py`, with GUI support in `app/sensitivity_dialog.py` and `app/main.py`.

Sensitivity analysis is a one-variable-at-a-time screening workflow. It builds a baseline scenario plus selected
parameter variations, runs the applicable static or transition response path, and ranks the effect on configured
engineering metrics. It is intended to highlight influential inputs and possible design-review focus areas, not to
replace multi-variable optimisation or a governing design envelope.

AS5100-specific handling keeps the train arrangement internally consistent:

- wheel-load sensitivity scales every axle in the AS5100 load set,
- AS5100 position sensitivity shifts every axle by the configured offset values,
- recommendation text treats AS5100 position changes as reference-position screening for the fixed arrangement.

## 16. Data Model and Engineering Dataset Scope

Primary schema: `db/models.py`.

Core tables include:

- reference properties:
  - `rails`,
  - `rail_steel_properties`,
  - `rail_admissible_stress`,
  - `rail_admissible_shear_stress`,
  - `sleepers`,
  - `pads`,
  - `support_profiles`,
  - `dynamic_track_parameters`,
  - `dipped_joint_reference_sets`.
- project and run context:
  - `projects`,
  - `track_configs`,
  - `load_cases`,
  - `results`.

This structure supports reproducible project setups and reference-set reuse.

## 17. Export and Reproducibility

Primary implementations: `core/exports.py`, `app/export_helpers.py`.

## 17.1 Static and envelope exports

- analysis CSV (`x, deflection, moment, shear, reaction` + stress columns),
- sleeper CSV (seat load, ballast pressure, ballast/capping signed + compressive columns),
- envelope CSV variants,
- transition metrics and transition series CSVs,
- transition run JSON.

## 17.2 Dynamic exports

- time-history CSV (`t, w, M, V, reaction, damping_force`),
- FFT CSV,
- PSD CSV,
- dynamic transition metrics/series CSV,
- dipped-joint CSV summary.

## 17.3 Metadata sidecars

Each export can have `*.meta.json` with:

- solver mode,
- SI units marker,
- UTC timestamp,
- SHA-256 hash of normalized input payload.

Transition/stress exports also include schema and semantics metadata fields in payload paths (for auditability).

AS5100 load-source metadata is also preserved in the applicable export metadata paths. This is important because
the same envelope output can represent either a fixed selected arrangement or a governing candidate selected by
sweep.

## 18. Chart Rendering and Label Controls

Primary implementation: `app/main.py`.

BOEF chart labels are part of the engineering traceability surface, but they can be visually filtered to reduce
plot clutter:

- `Inputs`: input/provenance badges and load-position marker labels.
- `Outputs`: result/KPI badges and result notes.
- `Max/min`: curve peak/minimum point labels and related critical point labels.

These controls apply across static, envelope, transition, dynamic, special, and stress chart render paths. They
only affect rendered label visibility; they do not alter solver arrays, summaries, exports, or metadata. Dynamic
charts also keep the separate `Dynamic annotations` setting for full/compact/off traceability detail.

For dense charts such as pressure and stress, the same input/output traceability can be rendered in a footer below
the plot rather than as text inside the axes. This is a presentation/layout choice only. The `Inputs` and `Outputs`
label controls still determine whether the footer receives provenance and KPI text.

## 19. Validation and Engineering Guardrails

Implemented checks include:

- positivity/non-negativity checks for physical parameters,
- domain-size checks relative to characteristic length (`1/beta`),
- singularity checks for the dynamic moving-frame denominator,
- compatibility checks for advanced dynamic options,
- profile and series length/monotonicity checks for transitions,
- explicit blocking of unsupported combinations.

Engineering sanity checks referenced in UI/workflows include:

- reaction equilibrium trend checks,
- symmetry behavior checks,
- derivative coherence checks.

## 20. Capabilities and Typical Engineering Applications

## 20.1 Typical use cases

- rail bending and deflection under wheel/axle sets,
- AS5100 fixed/governing envelope positioning for rail load cases,
- AS5100 reference-position screening through sensitivity scenarios,
- support demand and sleeper pressure screening,
- transition-zone stiffness-change comparison,
- envelope-based worst-case design positioning,
- dynamic amplification and frequency-domain behavior review,
- dipped-joint force screening,
- floating-slab isolation tuning.

## 20.2 Strengths

- multiple analysis fidelities in one tool,
- direct equation transparency via SI-based outputs,
- transition and stress post-processing structured for reporting,
- label controls that support cleaner presentation views without losing traceability,
- sensitivity screening that preserves load-set consistency for AS5100 and train arrangements,
- reproducible export metadata for technical governance.

## 20.3 Applicability limits

- model is analytical/numerical BOEF-focused, not full 3D FEM.
- some advanced combinations are intentionally constrained (for physical and numerical consistency).
- dynamic pressure outputs beyond rail bending stress are limited in this release.

## 21. Symbol Glossary

- `E`: elastic modulus (`Pa`)
- `I`: second moment of area (`m^4`)
- `A`: area (`m^2`)
- `G`: shear modulus or Pasternak shear term (context-dependent in module)
- `k`: foundation modulus (`N/m^2`)
- `k_g`: Pasternak shear foundation parameter
- `beta`: BOEF characteristic parameter (`1/m`)
- `w`: deflection (`m`)
- `M`: bending moment (`N·m`)
- `V`: shear force (`N`)
- `R_support`: rail support reaction per unit length (`N/m`), not ballast pressure
- `P`: point load (`N`)
- `S`: sleeper spacing (`m`)
- `Z`: section modulus (`m^3`)
- `sigma`: stress (`Pa`)
- `u`: energy density (`J/m`)
- `U`: integrated energy (`J`)
- `v`: speed (`m/s`)
- `f`: frequency (`Hz`)
- `PSD`: power spectral density (`m^2/Hz`)
- `x_ref` / `x0`: moving-load or AS5100 reference position, depending on context.

## 22. Source Map for Deep Dives

- Static closed-form equations: `core/model.py`
- Load builders and AS5100 metadata: `core/load_builder.py`
- Static run assembly + design checks: `core/analysis.py`
- Static numerical backend: `core/analysis_engine.py`
- Finite-difference solvers: `core/solver.py`
- Envelope analysis: `core/envelope.py`
- Transition metrics and energy: `core/transition.py`
- Stress/pressure post-processing: `core/stress_metrics.py`
- Dynamic config/solver/validation/engine: `core/dynamic/config.py`, `core/dynamic/solver.py`, `core/dynamic/validation.py`, `core/dynamic/engine.py`
- Special floating slab: `core/special/solver.py`
- Track modulus estimation: `core/track_modulus.py`
- Sensitivity scenarios and scoring: `core/sensitivity.py`, `app/sensitivity_dialog.py`
- Export and metadata pipeline: `core/exports.py`, `app/export_helpers.py`
- Engineering help text used in-app: `app/help_content.py`
- Chart rendering, labels, and GUI orchestration: `app/main.py`

## 23. Suggested Presentation Structure

For technical presentations, a practical sequence is:

1. Problem framing and BOEF assumptions.
2. Static theory and solver options.
3. AS5100, transition, and envelope decision workflow.
4. Dynamic analysis and critical-speed behavior.
5. Stress and pressure interpretation conventions.
6. Case studies with exported metrics and metadata.
7. Applicability limits and recommended validation checks.

This sequence aligns with how the application is organized and how engineering decisions are typically made.
