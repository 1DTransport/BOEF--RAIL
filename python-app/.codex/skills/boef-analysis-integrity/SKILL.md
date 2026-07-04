---
name: boef-analysis-integrity
description: Ensure BOEF GUI inputs propagate through analysis and design into solver with strict SI unit consistency, deterministic rebuilds per run, and auditable verification logging or tests; use when analysis results do not change after input edits, when GUI/design/solver/db code changes, or when standards-based material data is updated.
---

# BOEF Analysis Integrity & Engineering Verification

## Scope

Apply to GUI/controller layers, analysis input builders, solver invocation paths, design-check logic, database schema/seed scripts, and unit conversions at system boundaries.

## Required workflow

### 1. Trace full input data flow

- Identify each GUI widget for materials, geometry, loads, speeds, and design factors.
- Identify where each GUI value is read.
- Identify where each value is converted (including unit conversion).
- Identify where each value is stored in analysis input/config objects.
- Identify where each value is consumed by solver or design logic.
- Confirm the end-to-end path; do not assume propagation.

### 2. Verify analysis inputs rebuild on each run

- Confirm clicking “Run analysis” rebuilds a fresh analysis input object.
- Confirm no cached or mutated objects are reused.
- If stale data or reuse is detected, fix it and ensure immutability per run.

### 3. Verify design inputs separately from solver inputs

- Read design inputs explicitly from GUI.
- Inject them into a dedicated design input structure.
- Log them alongside solver inputs.
- Ensure design checks use these values deterministically.
- Do not allow silent defaults.

### 4. Enforce unit consistency

- Verify GUI units (mm, MPa, kN, km/h).
- Verify internal units (SI only).
- Verify solver inputs (SI only).
- Add logging that shows GUI value + unit and converted SI value.
- Flag and fix mixed units, hard-coded constants, and implicit conversions.

### 5. Database schema and seed verification

- Confirm required engineering properties exist.
- Allow optional fields only when justified.
- Use authoritative standards-based seed data; never guess.
- Confirm re-seeding behavior (idempotent vs destructive).

### 6. Add verification logging or tests

Add at least one of:

- Input verification logging showing GUI vs solver values.
- Minimal automated test that runs analysis with two different inputs and asserts different solver parameters.

### 7. UI diagnostics alignment (when charts/summaries change)

- If a sanity-check panel exists, ensure it is updated from the same result object used for plots.
- For chart thumbnail dashboards, refresh thumbnails only after a completed analysis run to avoid stale data.
- For transition workflows, keep summary metrics and chart overlays aligned:
  - transition annotation text on charts must reflect the same `TransitionRunResult` used by the summary panel.
  - do not mix live widget state into rendered transition metadata after run completion.
- For custom charts, preserve axis-family integrity:
  - allow static/transition spatial family mixing only when domains are spatially compatible.
  - keep time/frequency/dynamic-domain families isolated.

### 8. Constraints

- Never guess engineering values.
- Never hardcode material properties.
- Never bypass validation logic.
- Never hide missing inputs with defaults.
- Keep analysis deterministic and reproducible.

## Transition-energy and A3902 guardrails (mandatory when touched)

### Transition-energy (static transition workflow)

- Treat energy as additive post-processing only; do not alter solver governing equations.
- Keep `k(x)` meaning explicit in outputs:
  - `k_units = "N/m^2"`
  - `k_representation = "continuous_per_unit_length"` only when Winkler semantics are valid.
- Emit Winkler energy metrics only when the active static foundation model is Winkler.
  - For non-Winkler models, keep `energy_metrics` and `energy_series` unset and mark reaction law as model-dependent.
- Maintain naming discipline:
  - `u_*` fields are energy density (`J/m`)
  - `U_*` fields are integrated energy (`J`)
- Envelope energy must use max-absolute upper-bound construction:
  - `|M|_ub = max(|M_max|, |M_min|)`
  - `|w|_ub = max(|w_max|, |w_min|)`
- Derivative/window implementation must be robust on non-uniform grids:
  - one-sided boundaries for `du/dx`
  - clipped window integration at domain edges with effective window length reported.

### A3902 checks

- Prevent curve-factor double counting:
  - keep A3902 `P_SV` based on static wheel load before curve amplification,
  - apply curve influence once in the intended branch/factor only.
- Keep pressure helpers physically safe for standalone use:
  - validate ballast depth, sleeper width, and effective bearing length inputs explicitly.

### Minimum verification set when these areas change

- Unit tests for:
  - envelope max-absolute energy construction,
  - `u`/`U` units and non-negativity,
  - non-uniform derivative + clipped window behavior,
  - A3902 curved-case consistency (no double-count curve amplification),
  - standalone helper input validation,
  - transition chart overlays contain transition-summary KPIs on transition render paths,
  - custom-chart selection supports `transition_profile` with static/transition spatial series.

## Output requirements

- Explain the data flow clearly.
- List verified inputs.
- List fixes applied with code references.
- Provide evidence (logs or tests) that input propagation works.
