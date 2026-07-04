# BOEF Python App

BOEF is an open-source desktop application built with PySide6, SQLAlchemy, and Alembic. It provides track
response calculations for an infinite beam on an elastic (Winkler) foundation and renders plots + exports for
engineering review. The core also includes numerical solvers and track modulus estimation utilities for advanced
BOEF validation workflows.

BOEF is provided under the MIT License. Project information and author details are available at
https://www.1dtransport.com. It is engineering software, not a substitute for professional engineering judgement,
independent checking, project-specific design review, or compliance with applicable standards and authority
requirements.

Recent updates include:
- Rail profile seeding with verified dimensions/area/mass and upsert behavior for existing rows.
- Rail design checks (DAF, bending, shear) surfaced in the static Summary tab.
- Optional static multi-wheel load input for superposition.
- AS5100 rail loading for `300LA` and `150LA`, with fixed selected arrangement and governing envelope sweep
  modes. The governing sweep reuses the existing envelope engine and carries candidate/governing provenance into
  summaries, chart labels, transition envelope analysis, and exports.
- AS5100 sensitivity support: load-scale scenarios adjust every axle in the arrangement, and position scenarios
  shift the full AS5100 train by fixed offsets for reference-position screening.
- Axle-based load sources are now explicitly split to per-rail wheel loads before solving; manual single/multiple
  point loads remain interpreted as per-rail wheel loads.
- Transition Zones (Design Metrics) in Static analysis with templates/presets, k(x) profiles, metrics,
  optional quasi-static envelope mode, and dedicated transition exports.
- Help/interpretation content for plots and results (chart Help buttons).
- Quasi-static envelope reference position (x_ref) support for offset-based load movement.
- Granular chart label controls across Static, Dynamic, Transition, Special, and Stress charts:
  `Inputs`, `Outputs`, and `Max/min`.
- Optional **All-charts** dashboard view with cached thumbnails and click-to-expand back to the single chart.
- Long-run UX upgrades: **Cancel** button, runtime estimate for envelope/transition, validation hints,
  and a sanity-check panel (equilibrium, symmetry, derivative coherence).
- Dynamic advanced options (opt-in): moving-oscillator excitation, boundary policy (`zero_pad`/`periodic_wrap`),
  and optional irregularity forcing (profile or synthetic PSD) while keeping moving-load behavior as default for
  steady-state/time-history modes.
- Dynamic Transition mode (single/envelope) with screening/full-profile fidelity and dedicated exports.
- Dynamic Transition constraints now enforce equation-supported combinations in UI/validation:
  - `full_profile` uses moving-load + `zero_pad` + no irregularity + viscous damping only.
  - `screening` may use periodic wrap, irregularity, and hysteretic damping; response is uniform-`k1`.
  - Dynamic transition `k(x)` chart always shows the configured transition profile.
- Global per-chart **Custom chart** builder with up to 4 y-axes and same-domain interpolation.
- Export reproducibility metadata sidecars (`*.meta.json`) with solver mode, SI units, and hashed inputs.
- New **Stress** chart (post-processing): rail top/bottom fibre bending stress plus ballast/capping pressures
  where available, with ballast-thickness input, per-series visibility checkboxes, and append-only stress exports.
- Static pressure charts show ballast-top contact pressure, capping/below-ballast pressure, and A3902
  formation/subgrade pressure lines when those inputs are available.
- Transition chart overlays now surface transition-summary KPIs across transition render tabs in a transparent,
  presentation-friendly style.
- Pressure and Stress tabs can place traceability/output summaries in a below-chart footer to keep dense plots
  readable.
- Custom chart compatibility now supports combining transition profile `k(x)` with static/transition spatial
  series when x-domain families are spatially compatible.
- New baseline non-regression fixtures and tests in `tests/non_regression/`.

## Complete Product Guide (for presentations and handover)

This section is a full walkthrough of the BOEF desktop app from basic static analysis to advanced dynamic
and transition workflows. It is written as an operational and presentation-ready reference.

### What BOEF does

BOEF provides a practical railway track analysis environment that combines:
- fast static beam-on-elastic-foundation calculations for day-to-day engineering checks,
- numerical static solvers for advanced support models and non-uniform support profiles,
- transition-zone design metrics for stiffness changes,
- dynamic response analysis (spatial, time-history, FFT/PSD, and transition risk),
- specialized dipped-joint and floating-slab studies,
- exportable, reproducible engineering outputs for reporting and review.

### Main analysis modes

The app has three top-level modes:
- `Static`
- `Dynamic`
- `Special`

Within those modes, BOEF exposes multiple workflows:
- Static single-position analysis.
- Static quasi-static envelope analysis (closed-form and advanced numerical variants).
- AS5100 fixed selected and governing envelope sweep loading.
- Static Transition Zones (design metrics, single/envelope).
- Dynamic steady-state and time-history moving-load analysis.
- Dynamic Transition analysis (single/envelope, screening/full-profile fidelity).
- Dynamic dipped-joint wheel/rail force analysis.
- Special floating-slab isolation analysis.

## Feature Walkthrough (start to finish)

### 1. Static analysis (simple baseline workflow)

Use this when you need fast beam response, sleeper loads, pressure, and summary checks with minimal setup.

#### Core inputs

- `Rail`, `Sleeper`, `Pad`, and `Support profile` selections from the database.
- Load definition via one of:
  - single point load (`Point load`, `Load position`),
  - several wheel loads (superposition table),
  - train/axle load builder (axle load, bogie/axle spacing, reference position).
  - AS5100 rail loading (`300LA` or `150LA`) using fixed selected arrangement or governing envelope sweep.
- Load interpretation:
  - manual single and multiple loads are per-rail wheel loads,
  - train/axle builder and AS5100 values are total axle loads split internally to one rail line.
- `Sleeper spacing`.
- `Ballast thickness` (default `300 mm`) used for capping-top pressure via 2:1 spread post-processing.
- Design parameters:
  - track quality,
  - probability level,
  - speed,
  - wheel radius,
  - curve toggle,
  - steel tensile strength.

#### Outputs (charts)

- Deflection `y(x)`.
- Bending moment `M(x)`.
- Shear force `V(x)`.
- Rail support reaction `R_support(x)`: continuous Winkler support reaction acting on the rail, not ballast pressure.
- Sleeper seat loads `Q`.
- Sleeper-ballast contact pressures at the top of ballast.
- Depth pressure chart:
  - ballast top / sleeper contact pressure,
  - capping-top or below-ballast pressure from the 2:1 spread helper,
  - A3902 formation and subgrade reference lines where the design inputs exist.
- Stress chart:
  - Rail stress - top fibre (bending),
  - Rail stress - bottom fibre (bending),
  - ballast-top pressure,
  - capping-top pressure (2:1 spread, load-conserving).

#### Outputs (summary panel)

- Characteristic BOEF values (`β`, zero-moment/contraflexure distances).
- Peak values and locations for deflection, moment, shear, reaction, sleeper load, pressure.
- Rail base stress.
- Design checks:
  - DAF/bending/shear admissibility checks.
  - A3902-aligned quasi-static checks (`P_DV`, `Q_R`, ballast/formation/subgrade pressure where inputs exist).
- Sanity checks:
  - equilibrium,
  - symmetry,
  - moment-curvature coherence,
  - shear-gradient coherence.

### 2. Static quasi-static envelope analysis

Use this to scan a movement range and compute worst-case max/min envelopes.

#### Additional inputs

- `Reference position x_ref`.
- Movement range start/end (absolute via `x_ref + offsets`).
- Movement increment `Δx_ref`.
- Analysis domain start/end (or auto).
- Formation depths.
- Effective bearing width/length (manual or sleeper geometry based).
- Rail count for total-load interpretation.

#### Outputs

- Envelope charts (`max/min`) for:
  - deflection,
  - moment,
  - shear,
  - reaction,
  - sleeper loads,
  - ballast/formation pressures.
- Envelope summary with governing extremes.
- Optional two-rail envelope charts when enabled.

#### AS5100 envelope workflow

AS5100 rail loading is exposed as a load-source option rather than a separate solver.

- `Fixed selected arrangement` builds the selected AS5100 axle group arrangement directly.
- `Governing envelope sweep` runs the existing BOEF envelope engine over compact candidate arrangements and records
  the arrangement governing the envelope result.
- The solver load is the per-rail wheel load. For AS5100 axle loading this is half the axle load, and the chart
  load markers show both wheel load per rail and axle load for traceability.
- The summary panel, chart annotations, transition-envelope path, and exports carry AS5100 metadata such as
  standard, model, reference position `x0`, group count, group spacing, selected upper-bound arrangement, candidate
  count, and governing metric.
- AS5100 governing sweep is an envelope workflow. Dynamic analysis uses the selected fixed AS5100 arrangement as
  the moving load set; it does not perform an AS5100 governing candidate sweep.
- Sensitivity analysis can scale the full AS5100 axle set or shift every AS5100 axle by the configured position
  offsets, which is useful for checking whether the chosen reference position is governing.

#### Stress chart conventions (all analysis types)

- Internal naming:
  - `sigma_top_fiber_pa`,
  - `sigma_bottom_fiber_pa`.
- Rail bending stress:
  - `sigma_top = M / Z_top`,
  - `sigma_bottom = -M / Z_bottom`.
- Pressure sign convention:
  - positive = compression.
- Ballast pressure:
  - `q_ballast = Q / (B*L)`.
- Capping pressure (`h_b` = ballast thickness):
  - `Q = q_ballast * (B*L)`,
  - `q_capping = Q / ((B + 2h_b) * (L + 2h_b))`.
- Envelope stress uses conservative worst-case magnitudes for rail stresses (`max-absolute`) and
  compressive-only pressure curves for plotting.
- Dynamic mode shows **peak dynamic bending stress** (rail only); sleeper/capping pressures are
  intentionally marked unavailable.

### 3. Static Transition Zones (Design Metrics)

Use this for bridge approaches, slab-ballast transitions, stiff spots, and support-change studies.

#### Inputs

- Transition group enable.
- Template and preset selection.
- Run mode:
  - `Single position`,
  - `Worst-case envelope`.
- Transition movement controls (`x_ref`, range, step) for envelope mode.
- Transition domain start/end (or auto).
- Profile type:
  - uniform,
  - step,
  - linear ramp,
  - exponential ramp,
  - local stiff segment.
- Stiffness parameters:
  - `k1`,
  - `k2` (non-uniform profiles),
  - transition length `L_t` (ramp/exponential),
  - segment length `L_c` (segment profile).

#### Transition outputs

- `Transition k(x)` chart.
- `Transition metrics` summary:
  - `Δw(s)`, `Δw(1m)`,
  - `κ_max`, `M_max`,
  - bending energy proxy,
  - max reaction gradient,
  - max sleeper load.

#### Transition energy diagnostics (static Winkler post-processing)

- Energy density:
  - `u_r(x) = M(x)^2 / (2EI)`
  - `u_f(x) = 0.5 * k(x) * w(x)^2`
  - `u_total = u_r + u_f`
- Integrated energy:
  - `U_r = ∫u_r dx`,
  - `U_f = ∫u_f dx`,
  - `U_total = U_r + U_f`
- KPIs shown in UI:
  - `η = U_f/U_total`,
  - `u_max (J/m) @ x`,
  - `max |du/dx| (J/m²) @ x`.
- Transition chart tabs reuse transition KPIs as lightweight overlays for faster engineering review:
  - analysis metadata (template/preset/mode/profile, `k1/k2`, `Lt/Lc`, domain),
  - metric highlights (`Δw(s)`, `Δw(1m)`, `κmax`, `Mmax`, `Ub`, `|dp/dx|`, sleeper load),
  - energy highlights (`η`, `u_max`, `max|du/dx|`) with envelope/boundary notes where applicable.

Notes:
- Energy diagnostics are emitted for Winkler-consistent transition runs.
- Envelope energy is labeled as conservative upper-bound proxy.
- Non-Winkler transition runs keep energy fields disabled to avoid misinterpretation.
- The energy chart overlays `u_total` on the transition `k(x)` plot so stiffness changes and energy
  concentrations can be reviewed together.
- These diagnostics are a static beam-on-Winkler surrogate, not a full layer-by-layer finite-element strain-energy
  model.

### 4. Advanced static solver capabilities

Use this when baseline static assumptions are insufficient.

#### Foundation and beam options

- Foundation models:
  - Winkler,
  - series (railpad + trackbed),
  - sleeper-mass static equivalent.
- Beam theory:
  - Euler-Bernoulli,
  - Timoshenko.

#### Support and geometry options

- Nonuniform foundation profile `k(x)` (uniform/step/ramp).
- Discrete sleepers/pads.
- Two-rail coupled analysis with optional asymmetric right-rail loading.
- Pasternak shear layer `k_g`.

#### Advanced damping inputs (static context)

- Railpad and trackbed damping / loss factors for support-model reporting.
- Static solution remains zero-velocity consistent (damping captured for audit/reporting, not static deflection physics).

#### Additional outputs in static summary

- Equivalent support stiffness `k_eq`.
- Maximum railpad force and trackbed force.
- Maximum sleeper deflection (static equivalent).

### 5. Dynamic analysis (moving load)

Dynamic mode includes:
- `Steady-state moving load (travelling wave solution)`.
- `Time-history (explicit)`.

Both share the analytical dynamic beam-on-foundation backend and produce spatial plus probe-based outputs.

#### Dynamic inputs

- Dynamic speed.
- Dynamic damping model:
  - viscous (`c` or damping-ratio path),
  - hysteretic (`η`).
- Probe locations.
- Time window and sampling rate.
- Domain length and spatial resolution.
- PSD segment length and overlap.

#### Optional advanced dynamic inputs

- Excitation mode:
  - moving load,
  - moving oscillator.
- Boundary mode:
  - zero-pad,
  - periodic-wrap.
- Oscillator parameters (unsprung mass, suspension stiffness, suspension damping).
- Irregularity forcing:
  - measured profile (`x`, `z`),
  - synthetic PSD level + seed.

Notes:
- These advanced options apply directly to dynamic steady-state/time-history.
- In dynamic transition mode, available combinations depend on solver fidelity (see Section 6).

#### Dynamic outputs

- Spatial charts:
  - dynamic deflection,
  - dynamic moment,
  - dynamic shear,
  - dynamic reaction.
- Probe and frequency-domain charts:
  - deflection time history,
  - damping-force time history,
  - FFT amplitude,
  - Welch PSD,
  - support impedance magnitude.
- Dynamic summary:
  - peak deflection/moment/shear/reaction with governing location.

### 6. Dynamic Transition analysis

Dynamic transition extends dynamic analysis to stiffness transitions.

#### Inputs

- Dynamic transition profile type:
  - uniform,
  - step,
  - linear ramp,
  - exponential ramp,
  - local stiff segment.
- `k1`, `k2`, transition length, segment length.
- Run mode:
  - single position,
  - worst-case envelope (`x_ref` range and step).
- Solver fidelity:
  - `screening`,
  - `full_profile`.

#### Fidelity semantics and constraints

- `screening`:
  - Solves dynamic response using uniform support modulus `k1` (fast approximation).
  - May use periodic boundary mode, irregularity excitation, and hysteretic damping.
  - Still reports and plots the configured transition `k(x)` profile for design context.
- `full_profile`:
  - Solves non-uniform `k(x)` directly in the transition spatial solver.
  - Requires moving-load excitation, `zero_pad` boundary mode, no irregularity excitation, and viscous damping.
  - Unsupported combinations are prevented in GUI and rejected by validation.
- Transition input consistency:
  - `foundation_modulus_n_per_m2` is aligned to transition `k1` in transition config builds.

#### Outputs

- Dynamic transition profile chart `k(x)`.
- Dynamic summary + transition indicators:
  - transition fidelity,
  - governing `x_ref`,
  - risk index,
  - critical speed ratio,
  - dynamic amplification.
- Transition metrics/series exports for comparative studies and reporting.

### 7. Dipped joint mode (dynamic)

Use this mode for wheel/rail impact force calculations over dipped joints.

#### Inputs

- Static wheel load `P0`.
- Total dip angle `2α`.
- Speed.
- Hertzian stiffness.
- Unsprung mass.
- Effective/equivalent track masses (`mT1`, `mT2`).
- Equivalent track stiffness and damping.

#### Outputs

- Peak forces `P1` and `P2`.
- Dynamic amplification factors for `P1` and `P2`.
- Dedicated dipped-joint summary panel and CSV export.

### 8. Special mode: Floating slab isolation

Use this for slab isolator tuning and isolation effectiveness studies.

#### Inputs

- Slab mass.
- Isolator stiffness and damping.
- Static load.
- Frequency range and number of points.

#### Outputs

- Transmissibility vs frequency.
- Attenuation (dB) vs frequency.
- Summary:
  - natural frequency,
  - damping ratio,
  - static deflection.

## Input Reference (GUI inventory)

### Materials and project data

- Rails, sleepers, pads, support profiles.
- Projects, track configurations, load cases.
- In-app management dialogs for create/update/delete.

### Static controls

- Analysis type/static mode.
- Point/several/train load definitions.
- Sleeper spacing.
- Design parameters (track quality, probability, speed, wheel radius, curve factor, tensile strength).
- Envelope settings (movement, domain, depths, bearing geometry, rail count).
- Transition settings (template, preset, profile type, k-values, lengths, movement range).
- Advanced solver controls (foundation model, beam theory, nonuniform profile, discrete supports, two-rail, Pasternak).

### Dynamic controls

- Dynamic mode selector.
- Dynamic annotations selector (`Full traceability`, `Compact`, `Off`).
- Speed and damping settings.
- Probe/time/frequency settings.
- Dynamic transition profile and envelope controls.
- Dipped-joint parameter group.
- Dynamic advanced options (excitation, boundary, oscillator, irregularity).

### Special controls

- Floating slab parameter group (mass, stiffness, damping, static load, frequency sweep).

### Chart label controls

The `Chart labels` controls reduce plot clutter without removing engineering traceability from the code path:

- `Inputs`: input/provenance badges and load-position marker labels.
- `Outputs`: result/KPI badges and result notes.
- `Max/min`: point labels for plotted maxima/minima and selected zero/intersection labels.

The controls apply across static, envelope, transition, dynamic, special, and stress chart render paths. Changing
them re-renders the currently available result views without rerunning the analysis where cached result objects are
available.

Dense pressure and stress charts may render the same input/output traceability in a footer below the plot instead
of overlaying it on the axes. The label controls still govern whether those footer fields are populated.

### Sensitivity controls

Sensitivity analysis performs one-variable-at-a-time screening from the current static or transition setup. For
AS5100 load selections:

- wheel-load sensitivity scales every AS5100 axle consistently,
- AS5100 position sensitivity shifts the full train arrangement by the configured offsets,
- recommendation text treats AS5100 position changes as reference-position screening rather than a track-design
  modification.

This is a screening workflow only. It does not replace a governing envelope run or independent design review.

## Output Reference (charts and panels)

### Static chart tabs

- Deflection.
- Moment.
- Shear.
- Reaction.
- Sleeper loads.
- Pressures.
- Stress.
- Rail deflection (L/R, when available).
- Rail moment (L/R, when available).
- Summary.
- Transition `k(x)` and Transition metrics (when transition mode is active).

### Dynamic chart tabs

- Dynamic deflection.
- Dynamic moment.
- Dynamic shear.
- Dynamic rail support reaction.
- Dynamic damping.
- Dynamic time history.
- Dynamic FFT.
- Dynamic PSD.
- Dynamic impedance.
- Stress (rail bending only).
- Dynamic summary.
- Dipped joint summary (mode-dependent).

### Special chart tabs

- Floating slab transmissibility.
- Floating slab attenuation.
- Stress tab remains visible with `Not available` notice.
- Special summary.

## Export and Reporting Matrix

### Static exports

- `analysis.csv` (single-run x-series) with appended stress columns:
  - `sigma_top_fiber_pa`,
  - `sigma_bottom_fiber_pa`.
- `sleeper_loads.csv` (single-run sleeper outputs) with appended pressure columns:
  - `ballast_pressure_signed_pa`,
  - `ballast_pressure_comp_pa`,
  - `capping_pressure_signed_pa`,
  - `capping_pressure_comp_pa`.
- `analysis_config.json` (inputs + metadata hash).
- Per-file metadata sidecar: `*.meta.json` (includes stress assumptions fields).

### Static envelope exports

- `envelope_analysis.csv` with appended stress columns:
  - `sigma_top_fiber_ub_pa`,
  - `sigma_bottom_fiber_ub_pa`.
- `envelope_sleeper_loads.csv` with appended pressure columns:
  - `ballast_pressure_max_comp_pa`,
  - `capping_pressure_max_comp_pa`,
  - `capping_pressure_signed_max_pa`,
  - `capping_pressure_signed_min_pa`.
- `envelope_config.json`.
- Per-file metadata sidecar (includes stress assumptions fields).

### Transition exports

- `transition_metrics.csv`.
- `transition_series.csv`.
- `transition_run.json`.
- Schema and semantics metadata:
  - metrics/series schema versions,
  - `k_units`,
  - `k_representation`,
  - `foundation_reaction_law`.

### Dynamic exports

- `dynamic_time_history.csv`.
- `dynamic_fft.csv`.
- `dynamic_psd.csv`.
- Per-file metadata sidecar.

### Dynamic transition exports

- `dynamic_transition_metrics.csv`.
- `dynamic_transition_series.csv`.
- `dynamic_transition_run.json`.
- Per-file metadata sidecar.

### Dipped joint export

- `dipped_joint.csv`.
- Per-file metadata sidecar.

## Productivity and presentation features

- Per-chart Help buttons with interpretation text and formulas.
- Chart label category toggles (`Inputs`, `Outputs`, `Max/min`) for presentation cleanup without losing traceability.
- Single-chart and All-charts views for rapid reporting.
- Overlay mode for before/after comparisons.
- Custom chart builder:
  - combine up to 4 series on up to 4 y-axes,
  - same-domain family enforcement,
  - static/transition spatial compatibility for transition profile comparisons,
  - interpolation for compatible mixed sources.
- Runtime estimates and long-run warnings for envelope/transition runs.
- Background worker execution with cancel support for long analyses.
- Input snapshots, hashed metadata, and deterministic export payloads for reproducibility.

## Recommended presentation storyline

1. Start with static single-run baseline (deflection, moment, shear, reaction).
2. Show envelope mode to communicate worst-case movement effects.
3. Introduce Transition Zones and energy diagnostics for stiffness-change design.
4. Move to advanced static solver options (nonuniform, multilayer, two-rail).
5. Demonstrate dynamic steady-state/time-history and probe-domain outputs.
6. Present dynamic transition risk and governing `x_ref`.
7. Add dipped-joint and floating-slab specialized studies as focused use-cases.
8. Close with export/reproducibility workflow and metadata traceability.

## Requirements

- Python 3.11+

## Setup

```bash
cd python-app
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
python -m pip install -e '.[dev]'
```

## One-command runner (macOS/Linux)

Use the `./boef` helper script to create/reuse a virtualenv and install dependencies
only when `pyproject.toml` changes. This keeps run/test to a single command without
reinstalling on every invocation.

```bash
cd python-app
./boef run
```

Run tests the same way:

```bash
./boef test
```

Build the local macOS app bundle:

```bash
VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./scripts/build_macos_app.sh
```

The build output is `dist/BOEF.app`. The packaging script verifies the bundle executable, `Info.plist`, Alembic
migration resources, and the app icon. The package icon is `resources/icon-windowed.icns`, bundled as
`Contents/Resources/icon-windowed.icns` with `CFBundleIconFile=icon-windowed.icns`.

To replace a local installed app after a successful build:

```bash
ditto --norsrc --noextattr --noqtn --noacl dist/BOEF.app /Applications/BOEF.app
```

Replacing `/Applications/BOEF.app` may require elevated approval depending on the machine.

### Verified run steps (macOS)

If the default `.venv` is broken (common Qt "cocoa" plugin issue), run with a fresh
virtualenv path and keep using it for future runs:

```bash
cd python-app
VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef run
```

Notes:
- The first run installs dependencies and may take a few minutes.
- Subsequent runs reuse `.venv_run` and should start immediately.

## Verified commands (ran successfully)

These commands were executed successfully from the `python-app` directory.

```bash
VENV_DIR=.venv_py311 PYTHON_BIN=python3.11 ./boef core --help
VENV_DIR=.venv_py311 PYTHON_BIN=python3.11 ./boef core --load-newtons 1000 --stiffness-newtons-per-meter 2000
VENV_DIR=.venv_py311 PYTHON_BIN=python3.11 ./boef core a3902 --track-class 3 --static-wheel-load-n 100000 --speed-kmh 80 --confidence-limit-tc 1.0 --beta-per-m 0.9 --foundation-modulus-n-per-m2 80000000 --sleeper-spacing-m 0.65 --sleeper-width-m 0.25 --sleeper-length-m 2.5 --rail-centres-m 1.5 --ballast-depth-m 0.3 --fill-depth-m 0.2 --json
```

## Run commands

### macOS

Recommended: **create the venv once, then reuse it** (do not re-run `python -m venv` on top of an existing venv).

```bash
cd python-app
python3.11 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -e '.[dev]'

source .venv/bin/activate
python -m app
```

Run without activating the venv:

```bash
.venv/bin/python -m app
```

If `python` is not on your PATH, use the full interpreter path:

```bash
/opt/homebrew/bin/python3.11 -m app
```

### Windows (PowerShell)

```powershell
cd C:\path\to\BOEF\python-app
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

.\.venv\Scripts\Activate.ps1
python -m app
```

Run without activating the venv:

```powershell
.\.venv\Scripts\python.exe -m app
```

### Windows (cmd.exe)

```cmd
cd C:\path\to\BOEF\python-app
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

.\.venv\Scripts\activate.bat
python -m app
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'matplotlib'`

This means the current virtualenv is missing dependencies. Reinstall with:

```bash
cd python-app
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m app
```

### Qt platform plugin error on macOS

If you see `Could not find the Qt platform plugin "cocoa"`, this is usually a corrupted
virtual environment (not a code issue). A clean, deterministic reinstall fixes it.

Resolution (macOS):

1. Restart the terminal (clears cached environment state).
2. Remove the existing virtual environment:
   ```bash
   rm -rf .venv
   ```
3. Recreate the virtual environment using an explicit interpreter:
   ```bash
   /opt/homebrew/bin/python3.11 -m venv .venv
   source .venv/bin/activate
   ```
4. Reinstall the application:
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -e .
   ```
5. Clear macOS quarantine flags:
   ```bash
   xattr -dr com.apple.quarantine .venv
   ```
6. Run the application using the virtualenv's Python:
   ```bash
   python -m app
   ```

Notes:
- Use `python` (from the active virtualenv) consistently.
- The error is typically due to stale Qt plugin paths or macOS Gatekeeper restrictions.
- Reinstalling dependencies without recreating the virtualenv may not resolve it.

Important:
- Do **not** re-run `python -m venv .venv` on an existing virtualenv. If it breaks, delete the venv and
  recreate it. Re-running `venv` on top of a broken env does not repair PySide6 plugins.

Alternatively, use the helper script which bootstraps the virtualenv automatically:

```bash
cd python-app
./boef run
```

### Run the engineering core CLI

```bash
python -m core --load-newtons 1000 --stiffness-newtons-per-meter 2000
python -m core deflection --load-newtons 1000 --stiffness-newtons-per-meter 2000 --json
python -m core a3902 --track-class 3 --static-wheel-load-n 100000 --speed-kmh 80 --confidence-limit-tc 1.0 --beta-per-m 0.9 --foundation-modulus-n-per-m2 80000000 --sleeper-spacing-m 0.65 --sleeper-width-m 0.25 --sleeper-length-m 2.5 --rail-centres-m 1.5 --ballast-depth-m 0.3 --fill-depth-m 0.2 --json
```

### Initialize the database (SQLite)

```bash
python -m db --sqlite-path ./boef.sqlite
```

The CLI (and the GUI on startup) now run `alembic upgrade head` automatically, so existing
databases are migrated in-place before any sessions or seed data are created.

### Rail seed data behavior

The seed routine updates rail dimension fields (height/widths/head height/web thickness), area, and mass
to the verified values when rail names match. Existing rail inertia/section modulus values are preserved;
new rails seeded only with dimensions/area get a conservative rectangular-section approximation for Iy and
Wy so they remain usable in analysis.

Support profile labels use MN/m² for the continuous foundation modulus. The migration path renames legacy
`MN/m` default support-profile names to `MN/m²` so displayed names match the implemented `N/m²` internal units.

### Migrations (Alembic)

```bash
alembic -c alembic.ini revision -m "create tables"
alembic -c alembic.ini upgrade head
```

### Tests

```bash
VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef test
```

GUI tests are skipped by default in headless environments. To include them locally:

```bash
BOEF_ENABLE_GUI_TESTS=1 VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef test
```

On some macOS/Qt sandboxed environments, direct `QApplication` creation can abort before the test runner
can report a normal failure. In that case, use the non-GUI suite as the executable gate from the repo root:

```bash
python-app/.venv_run/bin/python -m pytest -q python-app/tests
```

## Advanced solver options (static)

The Advanced solver (numerical) now supports optional multilayer support models and a Timoshenko rail beam.
Defaults keep the legacy Winkler + Euler–Bernoulli path unchanged.

### Foundation model (Advanced solver only)
Choose one of:
- **Winkler (single-layer)**: legacy foundation modulus `k` (default).
- **Series (railpad + trackbed)**: two-layer series stiffness using
  `k_eq = (k_pad * k_bed) / (k_pad + k_bed)` with per-support inputs.
- **Sleeper-mass (3-layer)**: static equivalent of railpad + sleeper + trackbed (mass ignored in static).

Inputs use **per-support** units (kN/m, kN·s/m). Internally the solver converts to **per-length** using
`k_per_length = k_support / s` where `s` is sleeper spacing.

### Beam theory (Advanced solver only)
Choose one of:
- **Euler–Bernoulli**: legacy beam model (default).
- **Timoshenko**: includes shear deformation with inputs `ν` and `κ`.

Defaults:
- `ν = 0.3`, `κ = 0.4`
- Rail area `A` uses the selected rail profile; you can override if needed.

Numerical static outputs distinguish the related angular and reaction quantities:
- `slope_rad` is the rail deflection slope `dw/dx`.
- `rotation_rad` is the Timoshenko cross-section rotation `φ` when Timoshenko is selected.
- `shear_angle_rad` is `φ - dw/dx` for Timoshenko shear deformation.
- For Pasternak support, `reaction_n_per_m` is the total support reaction `k*w - k_g*w''`; numerical
  exports also expose the `k*w` and `-k_g*w''` components separately.

Note:
- **Pasternak shear layer (k_g)** is not supported with **Timoshenko** beam theory. The GUI will disable
  Pasternak when Timoshenko is selected, and the solver enforces this constraint.

### Damping (static + dynamic)
- **Static**: damping inputs are captured, logged, and exported for audit, but **do not** change the static
  solution (dashpots contribute zero at zero velocity).
- **Dynamic**: choose **Viscous** (coefficient or ζ ratio) or **Hysteretic** (loss factor η) for the
  foundation. Hysteretic damping applies to the foundation only (not rail EI).

UI walkthrough:
- **Static (Advanced solver)**: set `Foundation model`, then choose **Foundation damping model** and enter
  railpad/trackbed (and pad, if using discrete supports) damping or loss factors. These values are recorded
  in logs/snapshots but do not change static results.
- **Dynamic**: select **Damping model** under Dynamic inputs, then provide either `cₓ`/`ζ` (viscous) or `η`
  (hysteretic). Damping forces appear in the dynamic time-history tab, and support impedance is shown in
  the impedance tab.

### Outputs
When multilayer models are selected, the Summary panel includes:
- Equivalent support stiffness `k_eq`
- Max railpad/trackbed force (per length)
- Max sleeper deflection (static equivalent)

For full equations and references, see `docs/multilayer-and-timoshenko.md`.

## Transition Zones (Design Metrics)

The Static Analysis screen includes a **Transition Zones** group (checkable). It adds a design-focused
workflow for stiffness transitions without altering the core static solver physics.

Key features:
- **Templates + presets** for common transition scenarios with editable parameters.
- **k(x) profile builder**: uniform, step, linear ramp, exponential ramp, or stiff segment.
- **Run mode**: single-position or **quasi-static envelope** (loads treated as offsets from `x_ref`).
- **Metrics**: Δw(s), Δw(1 m), κ_max / M_max, bending energy proxy, reaction gradient, peak sleeper load.
- **Energy diagnostics (Winkler, static post-processing)**:
  - density: `u_r(x) = M(x)^2 / (2EI)`, `u_f(x) = 0.5 * k(x) * w(x)^2`, `u(x) = u_r + u_f`
  - integrals: `U_r = ∫u_r dx`, `U_f = ∫u_f dx`, `U_total = U_r + U_f`
  - summary KPIs: `η = U_f / U_total`, peak `u_max`, peak `|du/dx|`
  - windowed energy uses a clipped sliding window (target `ΔL = sleeper spacing`).
  - normalized export-only terms (when a reference load exists): `U_total / P_ref` and `u_max / P_ref`.
- **Exports**: `transition_metrics.csv`, `transition_series.csv`, and `transition_run.json`.

Notes:
- Non-uniform k(x) profiles use the **numerical** solver path; uniform profiles keep the closed-form path.
- Envelope results are computed over the movement range and reported as max/min curves.
- `k(x)` is treated as a continuous Winkler modulus in **N/m²** with reaction law
  `q_f(x) = k(x)w(x)` in **N/m**.
- Envelope energy fields are labelled as **upper-bound proxies** based on
  `max(|M_max|, |M_min|)` and `max(|w_max|, |w_min|)`.
- The envelope bending-energy proxy now uses the pointwise max-absolute moment envelope before integration,
  so governing positive and negative moments at different x-locations are not understated.
- For non-Winkler static foundation models, transition energy metrics are intentionally disabled and exports
  are marked `foundation_reaction_law="model-dependent (energy metrics disabled for non-Winkler)"`.
- Export naming follows strict energy notation: `u_*` = density (**J/m**), `U_*` = integrated energy (**J**).
- Boundary artifact flags are exported when peak `u` or peak `|du/dx|` occurs at domain edges.
- Transition exports now include schema metadata (`transition_metrics_schema_version=2` and series v2)
  plus `k` semantics and energy-theory fields (`energy_method`, `energy_equations`, `energy_scope`).

## Help & interpretation

Each chart includes a **Help** button that opens context-sensitive interpretation text covering:
- governing equations and unit conventions,
- envelope semantics (including `x_ref`),
- transition metrics and practical interpretation.

Chart labels are separate from Help content. The `Inputs`, `Outputs`, and `Max/min` controls change what is drawn
on the plot, while Help continues to provide interpretation text on demand.

## Chart view modes

The plot area supports two display modes:
- **Single** (default): existing tabbed charts, unchanged behavior.
- **All**: a dashboard grid of up to 8 thumbnail tiles (no live reparenting). Clicking a tile jumps back
  to the corresponding tab in Single view. Thumbnails are refreshed after an analysis run, not on resize.

## Unit conversion strategy

- **SI internally**: core calculations expect meters, newtons, pascals, and related SI-derived units.
- **Convert at the boundary**: UI inputs use convenience units (kN, mm, MPa, etc.) and are converted once
  on input (see `core/units.py` and UI field bindings in `app/main.py`). Outputs convert back once for display.
- **Database storage**: material properties and project configuration are stored in SI units so reports and
  exports stay consistent regardless of UI selection.

## Model assumptions and limitations

- **Infinite beam on Winkler foundation** with linear response and point loads for the primary closed-form path.
- **Superposition** is used for multiple wheel/point loads.
- **Static design checks**: the Summary tab includes Eisenmann DAF, bending/shear stresses, and pass/fail
  flags based on admissible limits. These are derived from rail design inputs and stored rail geometry.
- **A3902-aligned static metrics**: the design summary now also computes quasi-static values using the
  Appendix B formulations:
  - `phi = 1 + delta * eta * t_c` with `eta = 1` for `V <= 60 km/h`, else `eta = 1 + (V-60)/140`
  - `P_DV = P_SV * phi` (with `P_SV` from the base static wheel load; curve amplification is not applied twice)
  - `y_max = P_DV * beta / (2k)`
  - `Q_R = F1 * S * k * y_max`
  - `P_A = F2 * Q_R / (B * L_eff)` with `L_eff = l - g` when rail centres are supplied
  - optional `P_F` / `P_S` formation and subgrade pressures when ballast/fill depths are provided.
- **AS5100 rail loading**: AS5100 300LA/150LA vertical rail loading is implemented as a BOEF load-source/envelope
  workflow. Governing sweep mode reuses the existing envelope engine and records arrangement provenance; it does
  not change static solver equations or automatically apply a dynamic load allowance. Dynamic workflows can use
  the fixed AS5100 arrangement as a load source, but AS5100 governing sweep remains a static/envelope workflow.
- **Sleeper seat loads** integrate the continuous foundation reaction over a tributary sleeper length.
- **Single-rail formulation** is used in the primary GUI workflows; two-rail and discrete-support solvers are
  available in `core.solver` for engineering validation. The GUI now exposes these numerical capabilities
  behind an optional “Advanced solver (numerical)” toggle; the default workflow remains the closed-form
  Winkler path to preserve legacy behavior.
  - Advanced mode can enable nonuniform k(x) profiles, discrete sleeper supports, two-rail coupling, and
    Pasternak shear (kg). Keep the toggle off to reproduce legacy results.
- **Linear elasticity** only; non-linear ballast response, rail fastener nonlinearity, and dynamic effects are
  not modeled in the GUI path.
- **Stress chart scope**:
  - elastic bending stresses only,
  - no wheel/rail contact stress,
  - no ballast shear stress,
  - 2:1 spread is an engineering approximation for pressure diffusion.

## Foundation modulus `k`

The app uses a Winkler foundation modulus, **`k`**, stored and solved internally as **N/m²**. In the UI, this is
entered as the support profile “Foundation modulus” (MN/m²), converted to N/m², and stored on the support profile record.
Reports, charts, and CSV exports convert the internal value back to MN/m² for display.
During analysis, that value is passed as `foundation_modulus_n_per_m2` and used to derive the beam parameter
`beta` via:

```
β = (k / (4 E I))^(1/4)
```

where `E` is rail elastic modulus and `I` is rail second moment of area. This `beta` governs deflection, moment,
reaction, and sleeper load calculations in the core model.

## Capability matrix and verification

The BOEF capability inventory, gaps, and verification evidence are documented in
`BOEF_CAPABILITY_MATRIX.md`. Use it as the authoritative reference for what is
implemented (closed-form, numerical, and track modulus estimation) and which
checks validate each solver path.

## License

BOEF is licensed under the MIT License. See the repository root `LICENSE` file.
If you use BOEF in engineering work, research, publications, presentations, training material, or derivative
software, please acknowledge Mahan Yoldashkhan as the author and refer readers to https://www.1dtransport.com.
