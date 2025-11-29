#!/usr/bin/env python3
"""Command Line Interface for Mantice Model."""

import argparse
import sys
import os
from pathlib import Path


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Mantice Model: Superdiffusive Transport via Self-Organized Quaternionic Mantices"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Reproduce paper
    reproduce_parser = subparsers.add_parser(
        "reproduce", help="Reproduce paper results"
    )
    reproduce_parser.add_argument(
        "--figures", action="store_true", help="Generate figures"
    )
    reproduce_parser.add_argument(
        "--tables", action="store_true", help="Generate tables"
    )
    reproduce_parser.add_argument(
        "--all", action="store_true", help="Generate everything"
    )

    # Run simulation
    sim_parser = subparsers.add_parser("simulate", help="Run simulations")
    sim_parser.add_argument(
        "--system", choices=["turbulence", "railway"], required=True
    )
    sim_parser.add_argument("--nodes", type=int, default=1000)
    sim_parser.add_argument("--steps", type=int, default=1000)

    # Analyze data
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("--input", type=str, required=True)
    analyze_parser.add_argument("--output", type=str)

    args = parser.parse_args()

    if args.command == "reproduce":
        from reproduce_paper import main as reproduce_main

        reproduce_main()

    elif args.command == "simulate":
        if args.system == "turbulence":
            from examples.turbulence_simulation import run_turbulence_simulation

            run_turbulence_simulation()
        else:
            from examples.railway_optimization import run_railway_example

            run_railway_example()

    elif args.command == "analyze":
        print(f"Analyzing {args.input}")
        # Implementation here

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
