"""CLI entry point for the engineering core."""

from __future__ import annotations

import argparse
import json

from core.analysis import (
    ballast_contact_pressure_a3902,
    dynamic_vertical_wheel_load,
    eisenmann_dynamic_factor,
    eisenmann_track_condition_factor,
    eisenmann_velocity_dependent_factor,
    formation_pressure_a3902,
    max_deflection_single_load,
    rail_seat_load_from_deflection,
    subgrade_pressure_a3902,
    vqi_for_track_class,
)
from core.model import compute_deflection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BOEF engineering core CLI")
    parser.add_argument(
        "--load-newtons",
        type=float,
        help="Applied load in newtons (N) for legacy deflection mode",
    )
    parser.add_argument(
        "--stiffness-newtons-per-meter",
        type=float,
        help="Foundation stiffness in N/m for legacy deflection mode",
    )
    subparsers = parser.add_subparsers(dest="command")

    deflection_parser = subparsers.add_parser(
        "deflection",
        help="Compute a simple deflection from load and stiffness",
    )
    deflection_parser.add_argument(
        "--load-newtons",
        type=float,
        required=True,
        help="Applied load in newtons (N)",
    )
    deflection_parser.add_argument(
        "--stiffness-newtons-per-meter",
        type=float,
        required=True,
        help="Foundation stiffness in N/m",
    )
    deflection_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output",
    )

    a3902_parser = subparsers.add_parser(
        "a3902",
        help="Compute A3902 quasi-static factors and pressures",
    )
    vqi_group = a3902_parser.add_mutually_exclusive_group(required=True)
    vqi_group.add_argument(
        "--track-class",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Track class used to resolve VQI mapping",
    )
    vqi_group.add_argument(
        "--vqi",
        type=float,
        help="Direct VQI input",
    )
    a3902_parser.add_argument("--static-wheel-load-n", type=float, required=True, help="PSV in newtons")
    a3902_parser.add_argument("--speed-kmh", type=float, required=True, help="Train speed in km/h")
    a3902_parser.add_argument(
        "--confidence-limit-tc",
        type=float,
        default=1.0,
        help="A3902 confidence-limit factor tc",
    )
    a3902_parser.add_argument(
        "--beta-per-m",
        type=float,
        required=True,
        help="Beam parameter beta in 1/m",
    )
    a3902_parser.add_argument(
        "--foundation-modulus-n-per-m2",
        type=float,
        required=True,
        help="Foundation modulus k in N/m^2",
    )
    a3902_parser.add_argument(
        "--sleeper-spacing-m",
        type=float,
        required=True,
        help="Sleeper spacing S in m",
    )
    a3902_parser.add_argument(
        "--sleeper-width-m",
        type=float,
        required=True,
        help="Sleeper width B in m",
    )
    a3902_parser.add_argument(
        "--sleeper-length-m",
        type=float,
        required=True,
        help="Sleeper length L in m",
    )
    a3902_parser.add_argument(
        "--rail-centres-m",
        type=float,
        required=True,
        help="Rail centre spacing in m",
    )
    a3902_parser.add_argument(
        "--factor-of-safety-f1",
        type=float,
        default=1.25,
        help="A3902 sleeper load factor F1",
    )
    a3902_parser.add_argument(
        "--factor-of-safety-f2",
        type=float,
        default=1.0,
        help="A3902 pressure factor F2",
    )
    a3902_parser.add_argument(
        "--ballast-depth-m",
        type=float,
        help="Ballast depth in m for formation/subgrade pressure checks",
    )
    a3902_parser.add_argument(
        "--fill-depth-m",
        type=float,
        default=0.0,
        help="Fill depth in m for subgrade pressure checks",
    )
    a3902_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "deflection":
        deflection = compute_deflection(
            load_newtons=args.load_newtons,
            stiffness_newtons_per_meter=args.stiffness_newtons_per_meter,
        )
        if args.json:
            print(json.dumps({"deflection_m": deflection}))
        else:
            print(f"Deflection: {deflection:.6f} m")
        return 0

    if args.command == "a3902":
        if args.track_class is not None:
            vqi = vqi_for_track_class(args.track_class)
        else:
            if args.vqi is None:
                parser.error("a3902 requires --track-class or --vqi")
            vqi = args.vqi
        delta = eisenmann_track_condition_factor(vqi)
        eta = eisenmann_velocity_dependent_factor(args.speed_kmh)
        phi = eisenmann_dynamic_factor(delta, eta, args.confidence_limit_tc)
        pdv = dynamic_vertical_wheel_load(args.static_wheel_load_n, phi)
        y_max = max_deflection_single_load(
            pdv,
            args.foundation_modulus_n_per_m2,
            args.beta_per_m,
        )
        q_r = rail_seat_load_from_deflection(
            sleeper_spacing_m=args.sleeper_spacing_m,
            foundation_modulus_n_per_m2=args.foundation_modulus_n_per_m2,
            max_deflection_m=y_max,
            factor=args.factor_of_safety_f1,
        )
        p_a, effective_bearing_length = ballast_contact_pressure_a3902(
            rail_seat_load_n=q_r,
            sleeper_width_m=args.sleeper_width_m,
            sleeper_length_m=args.sleeper_length_m,
            rail_centres_m=args.rail_centres_m,
            factor_of_safety_f2=args.factor_of_safety_f2,
        )
        p_f = None
        p_s = None
        if args.ballast_depth_m is not None and args.ballast_depth_m > 0.0:
            p_f = formation_pressure_a3902(
                ballast_contact_pressure_pa=p_a,
                ballast_depth_m=args.ballast_depth_m,
                sleeper_width_m=args.sleeper_width_m,
                effective_bearing_length_m=effective_bearing_length,
            )
            p_s = subgrade_pressure_a3902(
                ballast_contact_pressure_pa=p_a,
                ballast_depth_m=args.ballast_depth_m,
                fill_depth_m=args.fill_depth_m,
                sleeper_width_m=args.sleeper_width_m,
                effective_bearing_length_m=effective_bearing_length,
            )
        result = {
            "vqi": vqi,
            "delta": delta,
            "eta": eta,
            "tc": args.confidence_limit_tc,
            "phi": phi,
            "p_sv_n": args.static_wheel_load_n,
            "p_dv_n": pdv,
            "y_max_m": y_max,
            "q_r_n": q_r,
            "p_a_pa": p_a,
            "p_f_pa": p_f,
            "p_s_pa": p_s,
            "effective_bearing_length_m": effective_bearing_length,
        }
        if args.json:
            print(json.dumps(result))
        else:
            print(f"VQI: {result['vqi']:.3f}")
            print(f"delta: {result['delta']:.6f}")
            print(f"eta: {result['eta']:.6f}")
            print(f"tc: {result['tc']:.6f}")
            print(f"phi: {result['phi']:.6f}")
            print(f"PSV: {result['p_sv_n']:.3f} N")
            print(f"PDV: {result['p_dv_n']:.3f} N")
            print(f"y_max: {result['y_max_m']:.9f} m")
            print(f"Q_R: {result['q_r_n']:.3f} N")
            print(f"P_A: {result['p_a_pa']:.3f} Pa")
            if result["p_f_pa"] is not None:
                print(f"P_F: {result['p_f_pa']:.3f} Pa")
            if result["p_s_pa"] is not None:
                print(f"P_S: {result['p_s_pa']:.3f} Pa")
            print(f"L_eff: {result['effective_bearing_length_m']:.6f} m")
        return 0

    if args.load_newtons is None or args.stiffness_newtons_per_meter is None:
        parser.error(
            "Provide legacy deflection args (--load-newtons and --stiffness-newtons-per-meter) "
            "or use a subcommand."
        )
    deflection = compute_deflection(
        load_newtons=args.load_newtons,
        stiffness_newtons_per_meter=args.stiffness_newtons_per_meter,
    )
    print(f"Deflection: {deflection:.6f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
