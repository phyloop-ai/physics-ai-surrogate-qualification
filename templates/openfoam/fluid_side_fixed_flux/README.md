# OpenFOAM Template: Fluid-Side Fixed-Flux Channel

This template is the first solver path for `PL-WP-002-CLEANROOM`.

## Physics Choice

The benchmark uses a documented fluid-side simplification instead of full two-region CHT. This template uses:

- incompressible steady flow solved with `simpleFoam`;
- passive temperature transport solved through OpenFOAM's scalar-transport function object;
- fixed-gradient wall temperature boundary as the fluid-side heat-flux proxy;
- laminar setting by default.

This is intentionally modest. The evaluation is about field prediction, calibrated uncertainty, OOD honesty, and fallback gating, not turbulence-model heroics.

The scalar field is temperature in Kelvin, not an arbitrary passive scalar. The clean-room config uses coolant-like thermal properties with elevated viscosity for a steady laminar benchmark, and records the heat-flux mapping in each generated `case.json`.

Positive wall heat flux means the wall heats the fluid. The generated fixed-gradient boundary uses:

```text
gradient = +q_w / k
```

where `q_w` is converted from `kW/m2` to `W/m2`, and `k` is `physics.thermal_conductivity_W_mK`.

## Status

This is a runnable-case draft until OpenFOAM is installed and the generated cases pass:

```bash
blockMesh
snappyHexMesh -overwrite
checkMesh
simpleFoam
foamRun -solver functions
```

Do not treat generated fields as evidence until convergence is logged in `evidence_pack/03_data/convergence_log.csv` and physical plausibility passes in `evidence_pack/02_metrics/physical_plausibility_summary.json`.

## Geometry

The case factory generates:

- a rectangular 2D channel background mesh;
- rectangular fin obstacle STL surfaces attached to the lower wall;
- one inlet, one outlet, top/bottom wall patches, separate front/back symmetry patches.

Implementation detail: the generated OpenFOAM mesh is a one-cell-thick quasi-2D case with
`symmetryPlane` front/back patches. `snappyHexMesh` requires a fully 3D mesh during snapping;
the mid-plane slice is the 2D field source for downstream resampling.

The case factory enforces:

```text
h_fin <= max_fin_height_fraction_of_channel * h_ch
```

The rule is configured in `configs/study.toml`.
