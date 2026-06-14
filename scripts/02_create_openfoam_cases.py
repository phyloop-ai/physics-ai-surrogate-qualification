#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import (
    PARAMETER_ORDER,
    ensure_evidence_folders,
    file_sha256,
    load_config,
    read_csv,
    study_paths,
    write_csv,
    write_json,
)


CASE_FACTORY_VERSION = "2026-06-12-case-factory-cleanroom-v14"


def mm(value: Any) -> float:
    return float(value)


def m_from_mm(value: float) -> float:
    return value * 1e-3


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.pool:
        rows = [row for row in rows if row["pool"] == args.pool]
    if args.case_id:
        selected = [row for row in rows if row["case_id"] == args.case_id]
        if not selected:
            raise SystemExit(f"Unknown case_id: {args.case_id}")
        return selected
    if args.limit is not None:
        return rows[: args.limit]
    if args.all:
        return rows
    raise SystemExit("Choose one of --case-id, --limit, or --all.")


def geometry_for(row: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    geom_cfg = config["geometry"]
    h_ch_mm = mm(row["h_ch"])
    h_fin_mm = mm(row["h_fin"])
    d_in_mm = mm(row["d_in"])
    n_fin = int(float(row["n_fin"]))
    length_mm = float(geom_cfg["channel_length_mm"])
    thickness_mm = float(geom_cfg["case_thickness_mm"])
    start_mm = float(geom_cfg["fin_region_start_mm"])
    end_mm = float(geom_cfg["fin_region_end_mm"])
    max_frac = float(geom_cfg["max_fin_height_fraction_of_channel"])

    valid = h_fin_mm <= max_frac * h_ch_mm
    inlet_width_mm = min(d_in_mm, h_ch_mm)
    inlet_y0_mm = 0.5 * (h_ch_mm - inlet_width_mm)
    inlet_y1_mm = inlet_y0_mm + inlet_width_mm
    pitch_mm = (end_mm - start_mm) / n_fin
    fin_width_mm = min(float(geom_cfg["max_fin_width_mm"]), pitch_mm * float(geom_cfg["fin_width_fraction_of_pitch"]))
    fin_boxes = []
    for idx in range(n_fin):
        center_x = start_mm + (idx + 0.5) * pitch_mm
        fin_boxes.append(
            {
                "index": idx,
                "x0_mm": center_x - 0.5 * fin_width_mm,
                "x1_mm": center_x + 0.5 * fin_width_mm,
                "y0_mm": 0.0,
                "y1_mm": h_fin_mm,
                "z0_mm": 0.0,
                "z1_mm": thickness_mm,
            }
        )

    dx_mm = float(geom_cfg["background_dx_mm"])
    mesh_nx = max(20, round(length_mm / dx_mm))
    mesh_ny = max(8, round(h_ch_mm / dx_mm))

    return {
        "valid": valid,
        "invalid_reason": "" if valid else "h_fin exceeds configured channel-clearance rule",
        "length_mm": length_mm,
        "height_mm": h_ch_mm,
        "thickness_mm": thickness_mm,
        "n_fin": n_fin,
        "requested_inlet_width_mm": d_in_mm,
        "effective_inlet_width_mm": inlet_width_mm,
        "inlet_y0_mm": inlet_y0_mm,
        "inlet_y1_mm": inlet_y1_mm,
        "inlet_width_saturated": d_in_mm > h_ch_mm,
        "fin_height_mm": h_fin_mm,
        "fin_width_mm": fin_width_mm,
        "fin_pitch_mm": pitch_mm,
        "fin_boxes": fin_boxes,
        "mesh_nx": mesh_nx,
        "mesh_ny": mesh_ny,
        "mesh_nz": 1,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def block_mesh_dict(geometry: dict[str, Any]) -> str:
    length = geometry["length_mm"]
    height = geometry["height_mm"]
    thick = geometry["thickness_mm"]
    nx = geometry["mesh_nx"]
    ny = geometry["mesh_ny"]
    inlet_y0 = geometry["inlet_y0_mm"]
    inlet_y1 = geometry["inlet_y1_mm"]

    y_candidates = [0.0, inlet_y0, inlet_y1, height]
    y_levels: list[float] = []
    for value in y_candidates:
        if not y_levels or abs(value - y_levels[-1]) > 1e-9:
            y_levels.append(value)

    vertices = []
    vertex_id: dict[tuple[int, int, int], int] = {}
    for z_idx, z_val in enumerate([0.0, thick]):
        for y_idx, y_val in enumerate(y_levels):
            for x_idx, x_val in enumerate([0.0, length]):
                vertex_id[(x_idx, y_idx, z_idx)] = len(vertices)
                vertices.append((x_val, y_val, z_val))

    def vid(x_idx: int, y_idx: int, z_idx: int) -> int:
        return vertex_id[(x_idx, y_idx, z_idx)]

    block_lines = []
    inlet_faces = []
    left_wall_faces = []
    outlet_faces = []
    bottom_faces = []
    top_faces = []
    front_faces = []
    back_faces = []
    for band_idx in range(len(y_levels) - 1):
        y_low = y_levels[band_idx]
        y_high = y_levels[band_idx + 1]
        dy = y_high - y_low
        band_ny = max(1, round(ny * dy / height))
        block_lines.append(
            f"    hex ({vid(0, band_idx, 0)} {vid(1, band_idx, 0)} {vid(1, band_idx + 1, 0)} {vid(0, band_idx + 1, 0)} "
            f"{vid(0, band_idx, 1)} {vid(1, band_idx, 1)} {vid(1, band_idx + 1, 1)} {vid(0, band_idx + 1, 1)}) "
            f"({nx} {band_ny} 1) simpleGrading (1 1 1)"
        )
        x0_face = f"({vid(0, band_idx, 0)} {vid(0, band_idx, 1)} {vid(0, band_idx + 1, 1)} {vid(0, band_idx + 1, 0)})"
        if abs(y_low - inlet_y0) < 1e-9 and abs(y_high - inlet_y1) < 1e-9:
            inlet_faces.append(x0_face)
        else:
            left_wall_faces.append(x0_face)
        outlet_faces.append(f"({vid(1, band_idx, 0)} {vid(1, band_idx + 1, 0)} {vid(1, band_idx + 1, 1)} {vid(1, band_idx, 1)})")
        front_faces.append(f"({vid(0, band_idx, 0)} {vid(0, band_idx + 1, 0)} {vid(1, band_idx + 1, 0)} {vid(1, band_idx, 0)})")
        back_faces.append(f"({vid(0, band_idx, 1)} {vid(1, band_idx, 1)} {vid(1, band_idx + 1, 1)} {vid(0, band_idx + 1, 1)})")

    bottom_faces.append(f"({vid(0, 0, 0)} {vid(1, 0, 0)} {vid(1, 0, 1)} {vid(0, 0, 1)})")
    top_idx = len(y_levels) - 1
    top_faces.append(f"({vid(0, top_idx, 0)} {vid(0, top_idx, 1)} {vid(1, top_idx, 1)} {vid(1, top_idx, 0)})")

    def render_faces(faces: list[str]) -> str:
        if not faces:
            return ""
        return "\n".join(f"            {face}" for face in faces)

    vertex_lines = "\n".join(f"    ({x:.9g} {y:.9g} {z:.9g})" for x, y, z in vertices)
    block_text = "\n".join(block_lines)
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 0.001;

vertices
(
{vertex_lines}
);

blocks
(
{block_text}
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
{render_faces(inlet_faces)}
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
{render_faces(outlet_faces)}
        );
    }}
    leftWall
    {{
        type wall;
        faces
        (
{render_faces(left_wall_faces)}
        );
    }}
    bottom
    {{
        type wall;
        faces
        (
{render_faces(bottom_faces)}
        );
    }}
    top
    {{
        type wall;
        faces
        (
{render_faces(top_faces)}
        );
    }}
    front
    {{
        type symmetryPlane;
        faces
        (
{render_faces(front_faces)}
        );
    }}
    back
    {{
        type symmetryPlane;
        faces
        (
{render_faces(back_faces)}
        );
    }}
);

mergePatchPairs
(
);
"""


def snappy_dict(geometry: dict[str, Any]) -> str:
    loc_x = m_from_mm(geometry["length_mm"] * 0.05)
    loc_y = m_from_mm(geometry["height_mm"] * 0.5)
    loc_z = m_from_mm(geometry["thickness_mm"] * 0.5)
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    fins
    {{
        type triSurfaceMesh;
        file "fins.stl";
    }}
}}

castellatedMeshControls
{{
    maxLocalCells 1000000;
    maxGlobalCells 2000000;
    minRefinementCells 0;
    nCellsBetweenLevels 2;

    features ();

    refinementSurfaces
    {{
        fins
        {{
            level (1 2);
            patchInfo {{ type wall; }}
        }}
    }}

    resolveFeatureAngle 30;
    refinementRegions {{}}
    locationInMesh ({loc_x:.9g} {loc_y:.9g} {loc_z:.9g});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
}}

meshQualityControls
{{
    maxNonOrtho 70;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-18;
    minTetQuality 1e-30;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

mergeTolerance 1e-6;
"""


def physics_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("physics", {})


def thermal_solver_mode(config: dict[str, Any]) -> str:
    return str(config["solver"].get("thermal_solver_mode", "passive_scalar_function"))


def common_system_files(config: dict[str, Any]) -> dict[str, str]:
    physics = physics_config(config)
    solver = config["solver"]
    nu = float(physics["kinematic_viscosity_m2_s"])
    dt = float(physics["thermal_diffusivity_m2_s"])
    scalar_n_correctors = int(solver.get("scalar_transport_n_correctors", 0))
    files = {
        "system/fvSchemes": fv_schemes_flow(config),
        "system/fvSchemes.flow": fv_schemes_flow(config),
        "system/fvSchemes.scalar": fv_schemes_scalar(config),
        "system/fvSchemes.thermal": fv_schemes_scalar(config),
        "system/fvSolution": fv_solution_flow(config),
        "system/fvSolution.flow": fv_solution_flow(config),
        "system/fvSolution.scalar": fv_solution_scalar(config),
        "system/fvSolution.thermal": fv_solution_thermal(config),
        "constant/transportProperties": f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}

transportModel Newtonian;
nu [0 2 -1 0 0 0 0] {nu:.9g};
DT [0 2 -1 0 0 0 0] {dt:.9g};
""",
        "constant/turbulenceProperties": """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}

simulationType laminar;
""",
        "system/surfaceFeatureExtractDict": """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeatureExtractDict;
}

fins.stl
{
    extractionMethod extractFromSurface;
    extractFromSurfaceCoeffs
    {
        includedAngle 150;
    }
    writeObj yes;
}
""",
    }
    if thermal_solver_mode(config) != "dedicated_energy_foam":
        files["system/functions"] = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      functions;
}}

#includeFunc scalarTransport(T, diffusivity=constant, D={dt:.9g}, nCorrectors={scalar_n_correctors})
"""
    return files


def fv_schemes_flow(config: dict[str, Any]) -> str:
    div_phi_u_scheme = str(config["solver"].get("flow_div_phi_u_scheme", "Gauss linearUpwind grad(U)"))
    return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) __DIV_PHI_U_SCHEME__;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""".replace("__DIV_PHI_U_SCHEME__", div_phi_u_scheme)


def fv_solution_flow(config: dict[str, Any]) -> str:
    solver = config["solver"]
    pressure_relaxation = float(solver.get("flow_pressure_relaxation", 0.3))
    velocity_relaxation = float(solver.get("flow_velocity_relaxation", 0.7))
    residual_target = float(solver.get("convergence_residual_target", 1e-6))
    return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    p { solver GAMG; tolerance 1e-7; relTol 0.01; smoother GaussSeidel; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.01; }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    residualControl
    {
        p __RESIDUAL_TARGET__;
        U __RESIDUAL_TARGET__;
    }
}

PIMPLE
{
    nCorrectors 1;
    nNonOrthogonalCorrectors 0;
    residualControl
    {
        p __RESIDUAL_TARGET__;
        U __RESIDUAL_TARGET__;
    }
}

relaxationFactors
{
    fields { p __PRESSURE_RELAXATION__; }
    equations { U __VELOCITY_RELAXATION__; }
}
""".replace("__RESIDUAL_TARGET__", f"{residual_target:.9g}").replace(
        "__PRESSURE_RELAXATION__", f"{pressure_relaxation:.9g}"
    ).replace("__VELOCITY_RELAXATION__", f"{velocity_relaxation:.9g}")


def fv_schemes_scalar(config: dict[str, Any]) -> str:
    ddt_scheme = str(config["solver"].get("scalar_transport_ddt_scheme", "Euler"))
    div_phi_t_scheme = str(config["solver"].get("scalar_div_phi_t_scheme", "bounded Gauss upwind"))
    return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes { default __DDT_SCHEME__; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,T) __DIV_PHI_T_SCHEME__;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""".replace("__DDT_SCHEME__", ddt_scheme).replace("__DIV_PHI_T_SCHEME__", div_phi_t_scheme)


def fv_solution_scalar(config: dict[str, Any]) -> str:
    relaxation = float(config["solver"].get("scalar_equation_relaxation", 0.7))
    return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    T
    {
        solver PBiCGStab;
        preconditioner DILU;
        tolerance 1e-7;
        relTol 0;
    }
}

PIMPLE
{
    nNonOrthogonalCorrectors 0;
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    residualControl
    {
        T 1e-6;
    }
}

relaxationFactors
{
    equations { T __SCALAR_RELAXATION__; }
}
""".replace("__SCALAR_RELAXATION__", f"{relaxation:.9g}")


def fv_solution_thermal(config: dict[str, Any]) -> str:
    relaxation = float(config["solver"].get("thermal_equation_relaxation", config["solver"].get("scalar_equation_relaxation", 0.7)))
    residual = float(config["solver"].get("thermal_residual_target", config["solver"].get("convergence_residual_target", 1e-6)))
    return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    T
    {
        solver PBiCGStab;
        preconditioner DILU;
        tolerance 1e-9;
        relTol 0;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    residualControl
    {
        T __THERMAL_RESIDUAL__;
    }
}

relaxationFactors
{
    equations { T __THERMAL_RELAXATION__; }
}
""".replace("__THERMAL_RELAXATION__", f"{relaxation:.9g}").replace("__THERMAL_RESIDUAL__", f"{residual:.9g}")


def control_dict_flow(config: dict[str, Any]) -> str:
    end_time = int(config["solver"].get("flow_end_time", 3000))
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""


def control_dict_scalar(config: dict[str, Any]) -> str:
    scalar_delta_t = float(config["solver"].get("scalar_transport_delta_t", 1e-4))
    scalar_duration = float(config["solver"].get("scalar_transport_duration_s", 0.05))
    scalar_write_interval = float(config["solver"].get("scalar_transport_write_interval_s", scalar_duration))
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

solver          functions;
subSolver       incompressibleFluid;
startFrom       latestTime;
startTime       0;
subSolverTime   0;
stopAt          endTime;
endTime         {scalar_duration:.12g};
deltaT          {scalar_delta_t:.12g};
writeControl    runTime;
writeInterval   {scalar_write_interval:.12g};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""


def control_dict_thermal(config: dict[str, Any]) -> str:
    solver = config["solver"]
    default_iterations = max(1, int(float(solver.get("scalar_transport_duration_s", 1))))
    iterations = max(1, int(solver.get("thermal_energy_iterations", default_iterations)))
    write_interval = max(1, int(solver.get("thermal_energy_write_interval", iterations)))
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     thermalEnergyFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {iterations};
deltaT          1;
writeControl    timeStep;
writeInterval   {write_interval};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""


def field_u(row: dict[str, str]) -> str:
    u_in = float(row["u_in"])
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}

dimensions [0 1 -1 0 0 0 0];
internalField uniform ({u_in:.9g} 0 0);
boundaryField
{{
    inlet {{ type fixedValue; value uniform ({u_in:.9g} 0 0); }}
    outlet {{ type zeroGradient; }}
    leftWall {{ type noSlip; }}
    top {{ type noSlip; }}
    bottom {{ type noSlip; }}
    fins {{ type noSlip; }}
    front {{ type symmetryPlane; }}
    back {{ type symmetryPlane; }}
}}
"""


def field_p() -> str:
    return """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    leftWall { type zeroGradient; }
    top { type zeroGradient; }
    bottom { type zeroGradient; }
    fins { type zeroGradient; }
    front { type symmetryPlane; }
    back { type symmetryPlane; }
}
"""


def field_t(row: dict[str, str], config: dict[str, Any]) -> str:
    physics = physics_config(config)
    inlet_temperature = float(physics["inlet_temperature_K"])
    conductivity = float(physics["thermal_conductivity_W_mK"])
    q_w = float(row["q_w"]) * 1000.0
    gradient = q_w / conductivity
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      T;
}}

dimensions [0 0 0 1 0 0 0];
internalField uniform {inlet_temperature:.9g};
boundaryField
{{
    inlet {{ type fixedValue; value uniform {inlet_temperature:.9g}; }}
    outlet {{ type zeroGradient; }}
    leftWall {{ type zeroGradient; }}
    top {{ type zeroGradient; }}
    bottom {{ type fixedGradient; gradient uniform {gradient:.9g}; }}
    fins {{ type fixedGradient; gradient uniform {gradient:.9g}; }}
    front {{ type symmetryPlane; }}
    back {{ type symmetryPlane; }}
}}
"""


def triangle_lines(vertices: list[tuple[float, float, float]]) -> list[str]:
    lines = ["  facet normal 0 0 0", "    outer loop"]
    for x, y, z in vertices:
        lines.append(f"      vertex {x:.12g} {y:.12g} {z:.12g}")
    lines.extend(["    endloop", "  endfacet"])
    return lines


def box_triangles(box: dict[str, float]) -> list[list[tuple[float, float, float]]]:
    x0, x1 = m_from_mm(box["x0_mm"]), m_from_mm(box["x1_mm"])
    y0, y1 = m_from_mm(box["y0_mm"]), m_from_mm(box["y1_mm"])
    z0, z1 = m_from_mm(box["z0_mm"]), m_from_mm(box["z1_mm"])
    v = {
        "000": (x0, y0, z0),
        "100": (x1, y0, z0),
        "110": (x1, y1, z0),
        "010": (x0, y1, z0),
        "001": (x0, y0, z1),
        "101": (x1, y0, z1),
        "111": (x1, y1, z1),
        "011": (x0, y1, z1),
    }
    return [
        [v["000"], v["100"], v["110"]], [v["000"], v["110"], v["010"]],
        [v["001"], v["011"], v["111"]], [v["001"], v["111"], v["101"]],
        [v["000"], v["001"], v["101"]], [v["000"], v["101"], v["100"]],
        [v["100"], v["101"], v["111"]], [v["100"], v["111"], v["110"]],
        [v["110"], v["111"], v["011"]], [v["110"], v["011"], v["010"]],
        [v["010"], v["011"], v["001"]], [v["010"], v["001"], v["000"]],
    ]


def stl_text(geometry: dict[str, Any]) -> str:
    lines = ["solid fins"]
    for box in geometry["fin_boxes"]:
        for tri in box_triangles(box):
            lines.extend(triangle_lines(tri))
    lines.append("endsolid fins")
    return "\n".join(lines) + "\n"


def allmesh_script() -> str:
    return """#!/bin/sh
set -eu

blockMesh
surfaceFeatureExtract || true
snappyHexMesh -overwrite
checkMesh
"""


def thermal_stability_script(config: dict[str, Any]) -> str:
    solver = config["solver"]
    max_delta = float(solver.get("scalar_steady_max_delta_T_K", 0.5))
    mean_delta = float(solver.get("scalar_steady_mean_delta_T_K", 0.05))
    min_writes = int(solver.get("scalar_steady_min_writes", 2))
    return f"""#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


INT_LINE_RE = re.compile(r"^\\d+$")


def foam_list_lines(text: str) -> list[str]:
    lines = text.splitlines()
    count_idx = None
    count = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if INT_LINE_RE.match(stripped):
            next_nonempty = next((item.strip() for item in lines[idx + 1 :] if item.strip()), "")
            if next_nonempty == "(":
                count_idx = idx
                count = int(stripped)
                break
    if count_idx is None or count is None:
        raise ValueError("Could not find OpenFOAM list count.")
    start_idx = next(idx for idx in range(count_idx + 1, len(lines)) if lines[idx].strip() == "(") + 1
    values = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == ")":
            break
        values.append(stripped)
        if len(values) == count:
            break
    if len(values) != count:
        raise ValueError(f"OpenFOAM list count mismatch: expected {{count}}, found {{len(values)}}.")
    return values


def parse_scalar_field(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"internalField\\s+uniform\\s+([^;]+);", text)
    if match:
        return [float(match.group(1))]
    return [float(line) for line in foam_list_lines(text)]


def numeric_time(path: Path) -> float | None:
    try:
        return float(path.name)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--after-time", type=float, required=True)
    parser.add_argument("--max-delta", type=float, default={max_delta:.12g})
    parser.add_argument("--mean-delta", type=float, default={mean_delta:.12g})
    parser.add_argument("--min-writes", type=int, default={min_writes})
    parser.add_argument("--output", default="postProcessing/thermal_stability.json")
    args = parser.parse_args()

    candidates = []
    for child in Path(".").iterdir():
        time_value = numeric_time(child)
        if time_value is None or time_value <= args.after_time + 1.0e-12:
            continue
        if (child / "T").exists():
            candidates.append((time_value, child / "T"))
    candidates.sort(key=lambda item: item[0])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {{
        "after_time": args.after_time,
        "candidate_write_count": len(candidates),
        "min_writes": args.min_writes,
        "mean_delta_T_K_limit": args.mean_delta,
        "max_delta_T_K_limit": args.max_delta,
        "passed": False,
    }}
    if len(candidates) < args.min_writes:
        payload["reason"] = "not enough scalar write times after flow solve"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(3)

    previous_time, previous_path = candidates[-2]
    current_time, current_path = candidates[-1]
    previous = parse_scalar_field(previous_path)
    current = parse_scalar_field(current_path)
    if len(previous) == 1 and len(current) > 1:
        previous = previous * len(current)
    if len(current) == 1 and len(previous) > 1:
        current = current * len(previous)
    if len(previous) != len(current):
        payload["reason"] = f"field-size mismatch: {{len(previous)}} vs {{len(current)}}"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(4)

    deltas = [abs(a - b) for a, b in zip(current, previous)]
    max_delta_observed = max(deltas) if deltas else 0.0
    mean_delta_observed = sum(deltas) / len(deltas) if deltas else 0.0
    passed = max_delta_observed <= args.max_delta and mean_delta_observed <= args.mean_delta
    payload.update(
        {{
            "previous_time": previous_time,
            "current_time": current_time,
            "window_s": current_time - previous_time,
            "cell_count": len(deltas),
            "max_delta_T_K": max_delta_observed,
            "mean_delta_T_K": mean_delta_observed,
            "passed": passed,
            "reason": "" if passed else "terminal scalar field changed beyond configured steady window",
        }}
    )
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if not passed:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
"""


def raw_outlet_enthalpy_script(config: dict[str, Any]) -> str:
    physics = config["physics"]
    solver = config["solver"]
    min_ratio = float(solver.get("raw_enthalpy_min_ratio", physics.get("energy_balance_min_ratio", -0.05)))
    max_ratio = float(solver.get("raw_enthalpy_max_ratio", physics.get("energy_balance_max_ratio", 10.0)))
    margin = float(solver.get("raw_enthalpy_abs_margin_K", physics.get("energy_balance_abs_margin_K", 0.0)))
    mass_balance_max = float(solver.get("raw_mass_balance_max_rel_error", 0.02))
    return """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


INT_LINE_RE = re.compile(r"^\\d+$")
VECTOR_RE = re.compile(r"\\(([^()]*)\\)")
FACE_RE = re.compile(r"^\\s*\\d+\\(([^()]*)\\)\\s*$")

MIN_RATIO = __MIN_RATIO__
MAX_RATIO = __MAX_RATIO__
ABS_MARGIN_K = __ABS_MARGIN_K__
MASS_BALANCE_MAX_REL_ERROR = __MASS_BALANCE_MAX_REL_ERROR__


def foam_list_lines(text: str) -> list[str]:
    lines = text.splitlines()
    count_idx = None
    count = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if INT_LINE_RE.match(stripped):
            next_nonempty = next((item.strip() for item in lines[idx + 1 :] if item.strip()), "")
            if next_nonempty == "(":
                count_idx = idx
                count = int(stripped)
                break
    if count_idx is None or count is None:
        raise ValueError("Could not find OpenFOAM list count.")
    start_idx = next(idx for idx in range(count_idx + 1, len(lines)) if lines[idx].strip() == "(") + 1
    values = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == ")":
            break
        values.append(stripped)
        if len(values) == count:
            break
    if len(values) != count:
        raise ValueError(f"OpenFOAM list count mismatch: expected {count}, found {len(values)}.")
    return values


def brace_block(text: str, name: str) -> str:
    match = re.search(r"(^|\\n)\\s*" + re.escape(name) + r"\\s*\\{", text)
    if not match:
        raise KeyError(f"Could not find block {name!r}.")
    start = match.end()
    depth = 1
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx]
    raise ValueError(f"Block {name!r} is not closed.")


def patch_info(boundary_text: str, patch_name: str) -> tuple[int, int]:
    block = brace_block(boundary_text, patch_name)
    n_faces = int(re.search(r"nFaces\\s+(\\d+)\\s*;", block).group(1))
    start_face = int(re.search(r"startFace\\s+(\\d+)\\s*;", block).group(1))
    return start_face, n_faces


def parse_label_list(path: Path) -> list[int]:
    return [int(line) for line in foam_list_lines(path.read_text(encoding="utf-8", errors="replace"))]


def parse_points(path: Path) -> list[tuple[float, float, float]]:
    points = []
    for line in foam_list_lines(path.read_text(encoding="utf-8", errors="replace")):
        match = VECTOR_RE.search(line)
        if not match:
            raise ValueError(f"Bad point row: {line}")
        x, y, z = (float(item) for item in match.group(1).split())
        points.append((x, y, z))
    return points


def parse_faces(path: Path) -> list[list[int]]:
    faces = []
    for line in foam_list_lines(path.read_text(encoding="utf-8", errors="replace")):
        match = FACE_RE.match(line)
        if not match:
            raise ValueError(f"Bad face row: {line}")
        faces.append([int(item) for item in match.group(1).split()])
    return faces


def parse_scalar_field(path: Path, n_cells: int) -> list[float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"internalField\\s+uniform\\s+([^;]+);", text)
    if match:
        return [float(match.group(1))] * n_cells
    values = [float(line) for line in foam_list_lines(text)]
    if len(values) != n_cells:
        raise ValueError(f"Scalar field count mismatch: {len(values)} values for {n_cells} cells.")
    return values


def parse_patch_scalar_values(path: Path, patch_name: str, n_faces: int) -> list[float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    boundary_text = text.split("boundaryField", 1)[1]
    block = brace_block(boundary_text, patch_name)
    uniform = re.search(r"value\\s+uniform\\s+([^;]+);", block)
    if uniform:
        return [float(uniform.group(1))] * n_faces
    nonuniform = re.search(r"value\\s+nonuniform\\s+List<scalar>\\s+(\\d+)\\s*\\((.*?)\\)\\s*;", block, re.S)
    if not nonuniform:
        raise ValueError(f"Could not parse patch values for {patch_name}.")
    values = [float(item) for item in nonuniform.group(2).split()]
    expected = int(nonuniform.group(1))
    if expected != n_faces or len(values) != n_faces:
        raise ValueError(f"Patch {patch_name} value count mismatch: expected {n_faces}, found {len(values)}.")
    return values


def vec_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def face_area(points: list[tuple[float, float, float]], face: list[int]) -> float:
    if len(face) < 3:
        return 0.0
    origin = points[face[0]]
    area = 0.0
    for idx in range(1, len(face) - 1):
        area += 0.5 * norm(cross(vec_sub(points[face[idx]], origin), vec_sub(points[face[idx + 1]], origin)))
    return area


def patch_area(boundary_text: str, points: list[tuple[float, float, float]], faces: list[list[int]], patch_name: str) -> float:
    start_face, n_faces = patch_info(boundary_text, patch_name)
    return sum(face_area(points, faces[idx]) for idx in range(start_face, start_face + n_faces))


def weighted_average(values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 1.0e-15:
        return float("nan")
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


def compute(case_dir: Path, flow_time: str, temperature_time: str) -> dict[str, object]:
    case_json = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    boundary_text = (case_dir / "constant" / "polyMesh" / "boundary").read_text(encoding="utf-8", errors="replace")
    owner = parse_label_list(case_dir / "constant" / "polyMesh" / "owner")
    points = parse_points(case_dir / "constant" / "polyMesh" / "points")
    faces = parse_faces(case_dir / "constant" / "polyMesh" / "faces")

    inlet_start, inlet_faces = patch_info(boundary_text, "inlet")
    outlet_start, outlet_faces = patch_info(boundary_text, "outlet")
    inlet_phi = parse_patch_scalar_values(case_dir / flow_time / "phi", "inlet", inlet_faces)
    outlet_phi = parse_patch_scalar_values(case_dir / flow_time / "phi", "outlet", outlet_faces)
    temperature = parse_scalar_field(case_dir / temperature_time / "T", max(owner) + 1)
    outlet_cells = owner[outlet_start : outlet_start + outlet_faces]
    outlet_t = [temperature[cell] for cell in outlet_cells]
    outlet_weights = [max(value, 0.0) for value in outlet_phi]

    inlet_inflow = sum(max(-value, 0.0) for value in inlet_phi)
    inlet_backflow = sum(max(value, 0.0) for value in inlet_phi)
    outlet_outflow = sum(outlet_weights)
    outlet_backflow = sum(max(-value, 0.0) for value in outlet_phi)
    net_inlet = -sum(inlet_phi)
    net_outlet = sum(outlet_phi)
    mass_balance_relative_error = abs(net_outlet - net_inlet) / max(abs(net_inlet), 1.0e-15)

    physics = case_json["physics_contract"]
    inlet_temperature = float(physics["inlet_temperature_K"])
    diffusivity = float(physics["thermal_diffusivity_m2_s"])
    wall_gradient = float(physics["wall_temperature_gradient_K_m"])
    heated_area = patch_area(boundary_text, points, faces, "bottom") + patch_area(boundary_text, points, faces, "fins")
    expected_delta = diffusivity * wall_gradient * heated_area / max(inlet_inflow, 1.0e-15)
    outlet_bulk = weighted_average(outlet_t, outlet_weights)
    outlet_delta = outlet_bulk - inlet_temperature if math.isfinite(outlet_bulk) else float("nan")
    balance_ratio = outlet_delta / expected_delta if expected_delta > 1.0e-12 and math.isfinite(outlet_delta) else float("nan")

    lower_bound = max(MIN_RATIO * expected_delta, -ABS_MARGIN_K)
    upper_bound = MAX_RATIO * expected_delta + ABS_MARGIN_K
    reasons = []
    if inlet_inflow <= 1.0e-15:
        reasons.append("no positive inlet inflow from raw phi")
    if outlet_outflow <= 1.0e-15:
        reasons.append("no positive outlet outflow from raw phi")
    if mass_balance_relative_error > MASS_BALANCE_MAX_REL_ERROR:
        reasons.append(
            f"raw inlet/outlet volume flow mismatch {mass_balance_relative_error:.6g} exceeds {MASS_BALANCE_MAX_REL_ERROR:.6g}"
        )
    if not math.isfinite(outlet_delta):
        reasons.append("nonfinite raw outlet bulk temperature")
    elif not (lower_bound <= outlet_delta <= upper_bound):
        reasons.append(
            f"raw outlet bulk temperature rise {outlet_delta:.6g} K outside enthalpy bounds "
            f"[{lower_bound:.6g}, {upper_bound:.6g}] K"
        )

    return {
        "balance_ratio": balance_ratio,
        "expected_delta_K": expected_delta,
        "flow_time": flow_time,
        "heated_area_m2": heated_area,
        "inlet_backflow_m3_s": inlet_backflow,
        "inlet_flow_m3_s": inlet_inflow,
        "inlet_temperature_K": inlet_temperature,
        "mass_balance_max_rel_error": MASS_BALANCE_MAX_REL_ERROR,
        "mass_balance_relative_error": mass_balance_relative_error,
        "net_inlet_flow_m3_s": net_inlet,
        "net_outlet_flow_m3_s": net_outlet,
        "outlet_backflow_m3_s": outlet_backflow,
        "outlet_bulk_K": outlet_bulk,
        "outlet_bulk_delta_K": outlet_delta,
        "outlet_flow_m3_s": outlet_outflow,
        "passed": not reasons,
        "ratio_bounds": [MIN_RATIO, MAX_RATIO],
        "reason": "; ".join(reasons),
        "temperature_time": temperature_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", default=".")
    parser.add_argument("--flow-time", required=True)
    parser.add_argument("--temperature-time", required=True)
    parser.add_argument("--output", default="postProcessing/raw_outlet_enthalpy.json")
    args = parser.parse_args()

    output = Path(args.output)
    try:
        payload = compute(Path(args.case_dir), args.flow_time, args.temperature_time)
    except Exception as exc:
        payload = {"passed": False, "reason": f"raw outlet enthalpy check error: {exc}"}
        write_payload(output, payload)
        raise SystemExit(6)

    write_payload(output, payload)
    if not payload["passed"]:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
""".replace("__MIN_RATIO__", f"{min_ratio:.12g}").replace(
        "__MAX_RATIO__",
        f"{max_ratio:.12g}",
    ).replace(
        "__ABS_MARGIN_K__",
        f"{margin:.12g}",
    ).replace(
        "__MASS_BALANCE_MAX_REL_ERROR__",
        f"{mass_balance_max:.12g}",
    )


def raw_temperature_extrema_script(config: dict[str, Any]) -> str:
    physics = config["physics"]
    inlet = float(physics["inlet_temperature_K"])
    min_allowed = max(
        float(physics["temperature_hard_min_K"]),
        inlet - float(physics["max_negative_delta_from_inlet_K"]),
    )
    max_allowed = min(
        float(physics["temperature_hard_max_K"]),
        inlet + float(physics["max_positive_delta_from_inlet_K"]),
    )
    return """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


INT_LINE_RE = re.compile(r"^\\d+$")
MIN_ALLOWED_K = __MIN_ALLOWED_K__
MAX_ALLOWED_K = __MAX_ALLOWED_K__


def foam_list_lines(text: str) -> list[str]:
    lines = text.splitlines()
    count_idx = None
    count = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if INT_LINE_RE.match(stripped):
            next_nonempty = next((item.strip() for item in lines[idx + 1 :] if item.strip()), "")
            if next_nonempty == "(":
                count_idx = idx
                count = int(stripped)
                break
    if count_idx is None or count is None:
        raise ValueError("Could not find OpenFOAM list count.")
    start_idx = next(idx for idx in range(count_idx + 1, len(lines)) if lines[idx].strip() == "(") + 1
    values = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == ")":
            break
        values.append(stripped)
        if len(values) == count:
            break
    if len(values) != count:
        raise ValueError(f"OpenFOAM list count mismatch: expected {count}, found {len(values)}.")
    return values


def parse_scalar_field(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"internalField\\s+uniform\\s+([^;]+);", text)
    if match:
        return [float(match.group(1))]
    return [float(line) for line in foam_list_lines(text)]


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    position = (len(sorted_values) - 1) * q / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", default=".")
    parser.add_argument("--temperature-time", required=True)
    parser.add_argument("--output", default="postProcessing/raw_temperature_extrema.json")
    args = parser.parse_args()

    output = Path(args.output)
    try:
        values = parse_scalar_field(Path(args.case_dir) / args.temperature_time / "T")
    except Exception as exc:
        payload = {"passed": False, "reason": f"raw temperature extrema check error: {exc}"}
        write_payload(output, payload)
        raise SystemExit(7)

    sorted_values = sorted(values)
    finite = all(math.isfinite(value) for value in values)
    t_min = min(values) if values else float("nan")
    t_max = max(values) if values else float("nan")
    t_mean = sum(values) / len(values) if values else float("nan")
    reasons = []
    if not values:
        reasons.append("empty raw temperature field")
    if not finite:
        reasons.append("nonfinite raw temperature value")
    if finite and t_min < MIN_ALLOWED_K:
        reasons.append(f"raw T_min_K {t_min:.6g} below allowed {MIN_ALLOWED_K:.6g}")
    if finite and t_max > MAX_ALLOWED_K:
        reasons.append(f"raw T_max_K {t_max:.6g} above allowed {MAX_ALLOWED_K:.6g}")

    payload = {
        "cell_count": len(values),
        "max_allowed_K": MAX_ALLOWED_K,
        "mean_T_K": t_mean,
        "min_allowed_K": MIN_ALLOWED_K,
        "min_T_K": t_min,
        "p01_T_K": percentile(sorted_values, 1),
        "p99_T_K": percentile(sorted_values, 99),
        "passed": not reasons,
        "reason": "; ".join(reasons),
        "temperature_time": args.temperature_time,
        "max_T_K": t_max,
    }
    write_payload(output, payload)
    if reasons:
        raise SystemExit(7)


if __name__ == "__main__":
    main()
""".replace("__MIN_ALLOWED_K__", f"{min_allowed:.12g}").replace(
        "__MAX_ALLOWED_K__",
        f"{max_allowed:.12g}",
    )


def allrun_script(config: dict[str, Any]) -> str:
    solver = config["solver"]
    mode = thermal_solver_mode(config)
    raw_enthalpy_enabled = bool(solver.get("raw_enthalpy_check_enabled", False))
    raw_temperature_extrema_enabled = bool(solver.get("raw_temperature_extrema_check_enabled", False))
    stability_enabled = bool(solver.get("scalar_steady_check_enabled", False))
    raw_enthalpy_block = ""
    if raw_enthalpy_enabled:
        raw_enthalpy_block = """
python3 tools/checkRawOutletEnthalpy.py --flow-time "$latest" --temperature-time "$temperature_latest" > log.rawOutletEnthalpy 2>&1 || raw_enthalpy_status=$?
"""
    raw_temperature_extrema_block = ""
    if raw_temperature_extrema_enabled:
        raw_temperature_extrema_block = """
python3 tools/checkRawTemperatureExtrema.py --temperature-time "$temperature_latest" > log.rawTemperatureExtrema 2>&1 || raw_temperature_extrema_status=$?
"""
    stability_block = ""
    if stability_enabled:
        stability_block = """
python3 tools/checkThermalStability.py --after-time "$latest" > log.thermalStability 2>&1 || thermal_stability_status=$?
"""
    if mode == "dedicated_energy_foam":
        thermal_block = """
cp system/controlDict.thermal system/controlDict
cp system/fvSchemes.thermal system/fvSchemes
cp system/fvSolution.thermal system/fvSolution
thermal_duration="$(foamDictionary -entry endTime -value system/controlDict)"
thermal_end="$(python3 - "$latest" "$thermal_duration" <<'PY'
import sys
latest = float(sys.argv[1])
duration = float(sys.argv[2])
print(f"{latest + duration:.12g}")
PY
)"
foamDictionary system/controlDict -entry endTime -set "$thermal_end"
thermalEnergyFoam -noFunctionObjects
"""
    else:
        thermal_block = """
cp system/controlDict.scalar system/controlDict
cp system/fvSchemes.scalar system/fvSchemes
cp system/fvSolution.scalar system/fvSolution
scalar_duration="$(foamDictionary -entry endTime -value system/controlDict)"
scalar_write_interval="$(foamDictionary -entry writeInterval -value system/controlDict)"
scalar_end="$(python3 - "$latest" "$scalar_duration" <<'PY'
import sys
latest = float(sys.argv[1])
duration = float(sys.argv[2])
print(f"{latest + duration:.12g}")
PY
)"
foamDictionary system/controlDict -entry endTime -set "$scalar_end"
foamDictionary system/controlDict -entry writeInterval -set "$scalar_write_interval"
foamRun -solver functions
"""
    return """#!/bin/sh
set -eu

./Allmesh
cp system/controlDict.flow system/controlDict
cp system/fvSchemes.flow system/fvSchemes
cp system/fvSolution.flow system/fvSolution
simpleFoam -noFunctionObjects

latest="$(foamListTimes -latestTime | tail -n 1)"
cp 0/T "$latest/T"
""" + thermal_block + """
temperature_latest="$(foamListTimes -latestTime | tail -n 1)"

raw_enthalpy_status=0
raw_temperature_extrema_status=0
thermal_stability_status=0
""" + raw_enthalpy_block + raw_temperature_extrema_block + stability_block + """
check_status=0
if [ "$raw_enthalpy_status" -ne 0 ]; then
    check_status="$raw_enthalpy_status"
fi
if [ "$raw_temperature_extrema_status" -ne 0 ] && [ "$check_status" -eq 0 ]; then
    check_status="$raw_temperature_extrema_status"
fi
if [ "$thermal_stability_status" -ne 0 ] && [ "$check_status" -eq 0 ]; then
    check_status="$thermal_stability_status"
fi
if [ "$check_status" -ne 0 ]; then
    exit "$check_status"
fi
"""


def write_case(case_dir: Path, row: dict[str, str], config: dict[str, Any], force: bool) -> dict[str, Any]:
    if case_dir.exists():
        if not force:
            raise SystemExit(f"Case directory already exists: {case_dir}. Use --force to replace selected cases.")
        shutil.rmtree(case_dir)
    geometry = geometry_for(row, config)
    if not geometry["valid"]:
        raise SystemExit(f"Invalid geometry for {row['case_id']}: {geometry['invalid_reason']}")

    write_text(case_dir / "system" / "blockMeshDict", block_mesh_dict(geometry))
    write_text(case_dir / "system" / "snappyHexMeshDict", snappy_dict(geometry))
    write_text(case_dir / "system" / "controlDict", control_dict_flow(config))
    write_text(case_dir / "system" / "controlDict.flow", control_dict_flow(config))
    write_text(case_dir / "system" / "controlDict.scalar", control_dict_scalar(config))
    write_text(case_dir / "system" / "controlDict.thermal", control_dict_thermal(config))
    for rel_path, content in common_system_files(config).items():
        write_text(case_dir / rel_path, content)
    write_text(case_dir / "0" / "U", field_u(row))
    write_text(case_dir / "0" / "p", field_p())
    write_text(case_dir / "0" / "T", field_t(row, config))
    write_text(case_dir / "constant" / "triSurface" / "fins.stl", stl_text(geometry))
    write_text(case_dir / "Allmesh", allmesh_script())
    write_text(case_dir / "Allrun", allrun_script(config))
    (case_dir / "Allmesh").chmod(0o755)
    (case_dir / "Allrun").chmod(0o755)
    if bool(config["solver"].get("scalar_steady_check_enabled", False)):
        write_text(case_dir / "tools" / "checkThermalStability.py", thermal_stability_script(config))
        (case_dir / "tools" / "checkThermalStability.py").chmod(0o755)
    if bool(config["solver"].get("raw_enthalpy_check_enabled", False)):
        write_text(case_dir / "tools" / "checkRawOutletEnthalpy.py", raw_outlet_enthalpy_script(config))
        (case_dir / "tools" / "checkRawOutletEnthalpy.py").chmod(0o755)
    if bool(config["solver"].get("raw_temperature_extrema_check_enabled", False)):
        write_text(case_dir / "tools" / "checkRawTemperatureExtrema.py", raw_temperature_extrema_script(config))
        (case_dir / "tools" / "checkRawTemperatureExtrema.py").chmod(0o755)

    physics = physics_config(config)
    solver = config["solver"]
    mode = thermal_solver_mode(config)
    wall_heat_flux_W_m2 = float(row["q_w"]) * 1000.0
    wall_temperature_gradient_K_m = wall_heat_flux_W_m2 / float(physics["thermal_conductivity_W_mK"])
    thermal_solver_label = "thermalEnergyFoam" if mode == "dedicated_energy_foam" else "foamRun -solver functions"
    solver_sequence = ["blockMesh", "snappyHexMesh -overwrite", "checkMesh", "simpleFoam -noFunctionObjects", thermal_solver_label]
    if bool(solver.get("raw_enthalpy_check_enabled", False)):
        solver_sequence.append("raw outlet enthalpy check")
    if bool(solver.get("raw_temperature_extrema_check_enabled", False)):
        solver_sequence.append("raw temperature extrema check")
    if bool(solver.get("scalar_steady_check_enabled", False)):
        solver_sequence.append("terminal scalar field stability check")
    case_payload = {
        "case_factory_version": CASE_FACTORY_VERSION,
        "case_id": row["case_id"],
        "pool": row["pool"],
        "parameters": {name: row[name] for name in PARAMETER_ORDER},
        "units": {
            "h_ch": "mm",
            "h_fin": "mm",
            "d_in": "mm",
            "u_in": "m/s",
            "q_w": "kW/m^2",
        },
        "physics_simplification": config["solver"]["physics_simplification"],
        "physics_contract": {
            "scalar_semantics": physics["scalar_semantics"],
            "scalar_solution_regime": physics.get("scalar_solution_regime", ""),
            "working_fluid": physics["working_fluid"],
            "inlet_temperature_K": physics["inlet_temperature_K"],
            "kinematic_viscosity_m2_s": physics["kinematic_viscosity_m2_s"],
            "thermal_diffusivity_m2_s": physics["thermal_diffusivity_m2_s"],
            "thermal_conductivity_W_mK": physics["thermal_conductivity_W_mK"],
            "positive_wall_heat_flux": physics["positive_wall_heat_flux"],
            "fixed_gradient_sign": physics["fixed_gradient_sign"],
            "wall_heat_flux_W_m2": wall_heat_flux_W_m2,
            "wall_temperature_gradient_K_m": wall_temperature_gradient_K_m,
        },
        "flow_solver_contract": {
            "flow_end_time": solver.get("flow_end_time", 3000),
            "convergence_residual_target": solver.get("convergence_residual_target", 1e-6),
            "div_phi_u_scheme": solver.get("flow_div_phi_u_scheme", "Gauss linearUpwind grad(U)"),
            "pressure_relaxation": solver.get("flow_pressure_relaxation", 0.3),
            "velocity_relaxation": solver.get("flow_velocity_relaxation", 0.7),
        },
        "scalar_solver_contract": {
            "thermal_solver_mode": mode,
            "thermal_solver": thermal_solver_label,
            "thermal_energy_iterations": solver.get("thermal_energy_iterations", ""),
            "thermal_energy_write_interval": solver.get("thermal_energy_write_interval", ""),
            "thermal_equation_relaxation": solver.get("thermal_equation_relaxation", solver.get("scalar_equation_relaxation", 0.7)),
            "thermal_residual_target": solver.get("thermal_residual_target", solver.get("convergence_residual_target", 1e-6)),
            "ddt_scheme": solver.get("scalar_transport_ddt_scheme", "Euler"),
            "div_phi_t_scheme": solver.get("scalar_div_phi_t_scheme", "bounded Gauss upwind"),
            "equation_relaxation": solver.get("scalar_equation_relaxation", 0.7),
            "n_correctors": solver.get("scalar_transport_n_correctors", 0),
            "delta_t_s": solver.get("scalar_transport_delta_t"),
            "duration_s": solver.get("scalar_transport_duration_s"),
            "write_interval_s": solver.get("scalar_transport_write_interval_s", solver.get("scalar_transport_duration_s")),
            "steady_check_enabled": bool(solver.get("scalar_steady_check_enabled", False)),
            "steady_min_writes": solver.get("scalar_steady_min_writes", ""),
            "steady_mean_delta_T_K": solver.get("scalar_steady_mean_delta_T_K", ""),
            "steady_max_delta_T_K": solver.get("scalar_steady_max_delta_T_K", ""),
            "raw_enthalpy_check_enabled": bool(solver.get("raw_enthalpy_check_enabled", False)),
            "raw_enthalpy_min_ratio": solver.get("raw_enthalpy_min_ratio", physics.get("energy_balance_min_ratio", -0.05)),
            "raw_enthalpy_max_ratio": solver.get("raw_enthalpy_max_ratio", physics.get("energy_balance_max_ratio", 10.0)),
            "raw_enthalpy_abs_margin_K": solver.get("raw_enthalpy_abs_margin_K", physics.get("energy_balance_abs_margin_K", 0.0)),
            "raw_mass_balance_max_rel_error": solver.get("raw_mass_balance_max_rel_error", 0.02),
            "raw_temperature_extrema_check_enabled": bool(solver.get("raw_temperature_extrema_check_enabled", False)),
            "raw_temperature_min_allowed_K": max(
                float(physics["temperature_hard_min_K"]),
                float(physics["inlet_temperature_K"]) - float(physics["max_negative_delta_from_inlet_K"]),
            ),
            "raw_temperature_max_allowed_K": min(
                float(physics["temperature_hard_max_K"]),
                float(physics["inlet_temperature_K"]) + float(physics["max_positive_delta_from_inlet_K"]),
            ),
        },
        "solver_sequence": solver_sequence,
        "geometry": geometry,
        "status": "generated_not_solved",
    }
    write_json(case_dir / "case.json", case_payload)
    write_json(case_dir / "constant" / "geometry.json", geometry)
    return {
        "case_json_sha256": file_sha256(case_dir / "case.json"),
        "stl_sha256": file_sha256(case_dir / "constant" / "triSurface" / "fins.stl"),
        "mesh_nx": geometry["mesh_nx"],
        "mesh_ny": geometry["mesh_ny"],
    }


def convergence_schema() -> dict[str, Any]:
    return {
        "required_columns": [
            "case_id",
            "pool",
            "case_dir",
            "solver",
            "mesh_cells",
            "residual_U_initial",
            "residual_U_final",
            "residual_p_initial",
            "residual_p_final",
            "residual_T_initial",
            "residual_T_final",
            "flow_converged_reported",
            "iterations_flow",
            "iterations_temperature",
            "scalar_stability_passed",
            "scalar_mean_delta_T_K",
            "scalar_max_delta_T_K",
            "scalar_stability_window_s",
            "scalar_stability_reason",
            "raw_enthalpy_passed",
            "raw_enthalpy_outlet_delta_T_K",
            "raw_enthalpy_expected_delta_T_K",
            "raw_enthalpy_balance_ratio",
            "raw_enthalpy_mass_balance_rel_error",
            "raw_enthalpy_reason",
            "raw_temperature_extrema_passed",
            "raw_temperature_min_T_K",
            "raw_temperature_mean_T_K",
            "raw_temperature_max_T_K",
            "raw_temperature_reason",
            "wall_time_sec",
            "converged",
            "dropped",
            "drop_reason",
            "openfoam_version",
        ],
        "rule": "Every requested case must appear exactly once. Pending/generated cases are not final evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--all", action="store_true", help="Generate all selected rows.")
    selector.add_argument("--limit", type=int, help="Generate the first N selected rows.")
    selector.add_argument("--case-id", help="Generate one case id.")
    parser.add_argument("--pool")
    parser.add_argument("--dry-run", action="store_true", help="Only write audit summary, not case folders.")
    parser.add_argument("--force", action="store_true", help="Replace selected generated case folders.")
    parser.add_argument("--output-dir", default="openfoam_cases")
    parser.add_argument(
        "--parameter-file",
        default="evidence_pack/03_data/parameters.csv",
        help="CSV of case parameters, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--manifest-file",
        default="evidence_pack/03_data/case_factory_manifest.csv",
        help="Case-factory manifest path, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--summary-file",
        default="evidence_pack/03_data/case_factory_summary.json",
        help="Case-factory summary path, relative to repo root unless absolute.",
    )
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)
    parameter_path = Path(args.parameter_file)
    if not parameter_path.is_absolute():
        parameter_path = paths.root / parameter_path
    rows = select_rows(read_csv(parameter_path), args)
    output_dir = paths.root / args.output_dir

    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        case_dir = output_dir / row["case_id"]
        geometry = geometry_for(row, config)
        record = {
            "case_id": row["case_id"],
            "pool": row["pool"],
            "case_dir": str(case_dir.relative_to(paths.root)),
            "valid_geometry": str(geometry["valid"]).lower(),
            "invalid_reason": geometry["invalid_reason"],
            "n_fin": row["n_fin"],
            "h_ch_mm": row["h_ch"],
            "h_fin_mm": row["h_fin"],
            "d_in_mm": row["d_in"],
            "u_in_m_per_s": row["u_in"],
            "q_w_kw_m2": row["q_w"],
            "mesh_nx": geometry["mesh_nx"],
            "mesh_ny": geometry["mesh_ny"],
            "effective_inlet_width_mm": geometry["effective_inlet_width_mm"],
            "inlet_width_saturated": str(geometry["inlet_width_saturated"]).lower(),
            "case_json_sha256": "",
            "stl_sha256": "",
            "status": "dry_run" if args.dry_run else "generated_not_solved",
        }
        if not args.dry_run:
            hashes = write_case(case_dir, row, config, args.force)
            record.update(hashes)
        manifest_rows.append(record)

    manifest_path = Path(args.manifest_file)
    if not manifest_path.is_absolute():
        manifest_path = paths.root / manifest_path
    summary_path = Path(args.summary_file)
    if not summary_path.is_absolute():
        summary_path = paths.root / summary_path
    fieldnames = [
        "case_id",
        "pool",
        "case_dir",
        "valid_geometry",
        "invalid_reason",
        "n_fin",
        "h_ch_mm",
        "h_fin_mm",
        "d_in_mm",
        "u_in_m_per_s",
        "q_w_kw_m2",
        "mesh_nx",
        "mesh_ny",
        "effective_inlet_width_mm",
        "inlet_width_saturated",
        "case_json_sha256",
        "stl_sha256",
        "status",
    ]
    write_csv(manifest_path, manifest_rows, fieldnames)
    write_json(paths.evidence_pack / "03_data" / "convergence_log_schema.json", convergence_schema())
    write_json(
        summary_path,
        {
            "case_factory_version": CASE_FACTORY_VERSION,
            "dry_run": args.dry_run,
            "selected_cases": len(rows),
            "output_dir": args.output_dir,
            "parameter_file": str(parameter_path.relative_to(paths.root)),
            "manifest": str(manifest_path.relative_to(paths.root)),
        },
    )
    print(f"{'Dry-run checked' if args.dry_run else 'Generated'} {len(rows)} case(s). Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
