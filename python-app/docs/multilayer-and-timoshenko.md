# Multilayer Foundation + Timoshenko Beam (Static Advanced Solver)

This document describes the static advanced-solver extensions for multilayer track support and the
Timoshenko rail beam option. All inputs and internal calculations use SI units.

## Multilayer foundation models

### 2-layer series model (railpad + trackbed)
Per-support stiffness values are combined in series:

```
1 / k_eq = 1 / k_pad + 1 / k_bed
k_eq = (k_pad * k_bed) / (k_pad + k_bed)
```

The solver converts per-support inputs to per-length values using sleeper spacing:

```
k_per_length = k_support / s
```

### 3-layer sleeper-mass model (static equivalent)
For static analysis, sleeper mass does not influence deflection. The model is treated as a
static equivalent of the series model and still reports:
- railpad force (per length)
- trackbed force (per length)
- sleeper deflection (static equivalent)

This keeps the path compatible with the existing static solver while preserving layer reporting.

### Transition energy scope

Transition energy diagnostics are currently emitted only for Winkler-consistent static transition runs. Multilayer,
series, Pasternak, discrete-support, and other model-dependent foundation paths keep `energy_metrics` and
`energy_series` unset so the app does not present Winkler spring energy as layer-by-layer foundation energy.
When energy results are available, the equations are `u_rail = M^2/(2EI)`, `u_foundation = 0.5*k(x)*w^2`, and
`U = int(u) dx`.

### Chart and export presentation

Chart label visibility controls are presentation controls only. They can hide or show input labels, output labels,
and max/min labels across static, dynamic, transition, special, and stress plots, but they do not change the
solver path, stress recovery, or transition-energy eligibility. In particular, hiding a label must not be treated
as disabling the corresponding calculation, and enabling labels must not expose Winkler-only energy diagnostics
for multilayer, Pasternak, or discrete-support runs.

Continuous foundation stiffness values are displayed and exported as MN/m² while remaining N/m² internally. This
applies to equivalent support stiffness, non-uniform `k(x)` profiles, transition `k1/k2`, and impedance-style
dynamic chart labels. Discrete spring inputs such as pad stiffness remain per-support spring values and should not
be confused with continuous modulus units.

## Timoshenko beam option

The Timoshenko beam uses two fields:
- deflection `w(x)`
- rotation `φ(x)`

Static governing relations:
```
EI * d²φ/dx² = κGA * (φ - dw/dx)
V = κGA * (φ - dw/dx)
```

and vertical equilibrium with foundation reaction:
```
dV/dx + k * w(x) = q(x)
```

BOEF reports three separate angular quantities when available:
- `slope_rad = dw/dx` (Euler-Bernoulli/Pasternak rail slope).
- `rotation_rad = φ` (Timoshenko cross-section rotation; equal to slope only in the Euler-Bernoulli limit).
- `shear_angle_rad = φ - dw/dx` (Timoshenko shear deformation measure).

For Pasternak support, BOEF reports the total line reaction as:
```
reaction_n_per_m = k * w - k_g * d²w/dx²
```

This is split in numerical outputs as `winkler_reaction_n_per_m = k*w` and
`pasternak_shear_reaction_n_per_m = -k_g*d²w/dx²` so beam shear `V` is not confused with the
foundation shear-layer contribution.

Defaults (editable in the GUI):
- Poisson’s ratio `ν = 0.3`
- Shear correction factor `κ = 0.4`

The shear modulus is computed from `E` and `ν`:
```
G = E / (2(1+ν))
```

The Timoshenko solver is kept Winkler-only in this release. Pasternak + Timoshenko remains blocked until a
separate coupled formulation is implemented and benchmarked.

## References
- Lamprea-Pineda et al. (2022) BOEF model review: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9602678/
- Kerr foundation overview: https://onlinelibrary.wiley.com/doi/10.1002/9780470015902.a0000964.pub2
- Pasternak foundation example: https://www.semanticscholar.org/paper/5c8404ac9d7133928a5fc5-5daaba542cb5
- Timoshenko beam insight PDF: https://scispace.com/pdf/physical-insight-into-timoshenko-beam-theory-and-its-3uhs0ps3jo.pdf
