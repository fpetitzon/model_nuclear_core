#!/usr/bin/env python3
"""
Example PWR Core Simulation

This script demonstrates how to use the nuclear_core package
to simulate a 3000 MW Pressurized Water Reactor core.

Usage:
    python run_simulation.py [--power POWER] [--enrichment ENRICHMENT]

Example:
    python run_simulation.py --power 3000 --enrichment 4.0
"""

import argparse
import sys
import os

# Add parent directory to path for importing nuclear_core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuclear_core.reactor import PWRCore, create_pwr_core
from nuclear_core.materials import UO2Fuel, LightWaterModerator, FuelAssembly
from nuclear_core.geometry import CoreGeometry
from nuclear_core.neutronics import CriticalityCalculator, NeutronicsModel
from nuclear_core.thermal import ThermalHydraulics


def run_basic_simulation(power_mw: float = 3000.0, enrichment: float = 4.0):
    """
    Run a basic reactor core simulation.

    Args:
        power_mw: Thermal power in MW
        enrichment: U-235 enrichment in weight percent
    """
    print("\n" + "="*70)
    print("       NUCLEAR REACTOR CORE SIMULATION")
    print("       3000 MW Pressurized Water Reactor")
    print("="*70)

    # Create reactor core model
    print("\nInitializing reactor core model...")
    reactor = create_pwr_core(
        power_mw=power_mw,
        enrichment=enrichment,
    )

    # Run full analysis and print summary
    reactor.print_summary()

    return reactor


def run_parametric_study():
    """
    Perform a parametric study varying enrichment.

    Studies the effect of enrichment on criticality.
    """
    print("\n" + "="*70)
    print("       PARAMETRIC STUDY: EFFECT OF ENRICHMENT")
    print("="*70)

    enrichments = [3.0, 3.5, 4.0, 4.5, 5.0]

    print(f"\n{'Enrichment [%]':>15} {'k_infinity':>12} {'k_effective':>12} {'ρ [pcm]':>12}")
    print("-" * 55)

    for enrich in enrichments:
        reactor = create_pwr_core(power_mw=3000.0, enrichment=enrich)
        crit = reactor.calculate_criticality()

        k_inf = crit['four_factor']['k_infinity']
        k_eff = crit['six_factor']['k_effective']
        rho_pcm = crit['reactivity']['pcm']

        print(f"{enrich:>15.1f} {k_inf:>12.4f} {k_eff:>12.4f} {rho_pcm:>12.1f}")


def run_flux_calculation():
    """
    Calculate and display neutron flux distribution.
    """
    print("\n" + "="*70)
    print("       NEUTRON FLUX CALCULATION")
    print("="*70)

    reactor = create_pwr_core(power_mw=3000.0, enrichment=4.0)

    flux_data = reactor.calculate_neutron_flux()

    print(f"\nNeutron Flux Characteristics:")
    print(f"  Average thermal flux:   {flux_data['average_flux_n_cm2_s']:.3e} n/cm²/s")
    print(f"  Peak thermal flux:      {flux_data['peak_flux_n_cm2_s']:.3e} n/cm²/s")
    print(f"  Thermal/Fast ratio:     {flux_data['thermal_to_fast_ratio']:.3f}")
    print(f"  Migration length:       {flux_data['migration_length_cm']:.2f} cm")

    peaking = flux_data['peaking_factors']
    print(f"\nPeaking Factors:")
    print(f"  Radial:                 {peaking['radial']:.3f}")
    print(f"  Axial:                  {peaking['axial']:.3f}")
    print(f"  Total:                  {peaking['total']:.3f}")


def run_criticality_breakdown():
    """
    Detailed criticality calculation with all factors.
    """
    print("\n" + "="*70)
    print("       DETAILED CRITICALITY ANALYSIS")
    print("="*70)

    reactor = create_pwr_core(power_mw=3000.0, enrichment=4.0)
    crit = reactor.calculate_criticality()

    print("\n  FOUR-FACTOR FORMULA (Infinite Medium)")
    print("  " + "-" * 50)
    four = crit['four_factor']
    print(f"    η (reproduction factor):        {four['eta']:.4f}")
    print(f"      └─ Average neutrons per thermal absorption in fuel")
    print(f"    ε (fast fission factor):        {four['epsilon']:.4f}")
    print(f"      └─ Enhancement from U-238 fast fissions")
    print(f"    p (resonance escape prob.):     {four['p']:.4f}")
    print(f"      └─ Probability of escaping U-238 resonances")
    print(f"    f (thermal utilization):        {four['f']:.4f}")
    print(f"      └─ Fraction of thermal absorptions in fuel")
    print(f"\n    k_∞ = η × ε × p × f =           {four['k_infinity']:.4f}")

    print("\n  SIX-FACTOR FORMULA (Finite Reactor)")
    print("  " + "-" * 50)
    six = crit['six_factor']
    print(f"    P_F (fast non-leakage):         {six['P_fast']:.4f}")
    print(f"      └─ Probability fast neutrons don't leak")
    print(f"    P_T (thermal non-leakage):      {six['P_thermal']:.4f}")
    print(f"      └─ Probability thermal neutrons don't leak")
    print(f"\n    k_eff = k_∞ × P_F × P_T =       {six['k_effective']:.4f}")

    print("\n  REACTIVITY")
    print("  " + "-" * 50)
    rho = crit['reactivity']
    print(f"    ρ = (k-1)/k:                    {rho['delta_k_over_k']:.6f}")
    print(f"    ρ in pcm:                       {rho['pcm']:.1f}")
    print(f"    ρ in dollars:                   {rho['dollars']:.2f} $")

    print("\n  GEOMETRIC BUCKLING")
    print("  " + "-" * 50)
    buck = crit['buckling']
    print(f"    B²_r (radial):                  {buck['radial_cm2']:.6f} cm⁻²")
    print(f"    B²_z (axial):                   {buck['axial_cm2']:.6f} cm⁻²")
    print(f"    B²_total:                       {buck['total_cm2']:.6f} cm⁻²")


def run_thermal_analysis():
    """
    Detailed thermal-hydraulic analysis.
    """
    print("\n" + "="*70)
    print("       THERMAL-HYDRAULIC ANALYSIS")
    print("="*70)

    reactor = create_pwr_core(power_mw=3000.0, enrichment=4.0)
    th = reactor.calculate_thermal_hydraulics()

    print("\n  BULK COOLANT CONDITIONS")
    print("  " + "-" * 50)
    bulk = th['bulk_conditions']
    print(f"    Inlet temperature:              {bulk['inlet_temp_K']:.1f} K ({bulk['inlet_temp_K']-273.15:.1f} °C)")
    print(f"    Outlet temperature:             {bulk['outlet_temp_K']:.1f} K ({bulk['outlet_temp_K']-273.15:.1f} °C)")
    print(f"    Temperature rise:               {bulk['temp_rise_K']:.1f} K")
    print(f"    Mass flow rate:                 {bulk['mass_flow_kg_s']:.1f} kg/s")
    print(f"    Coolant velocity:               {bulk['coolant_velocity_m_s']:.2f} m/s")

    print("\n  HEAT TRANSFER")
    print("  " + "-" * 50)
    ht = th['heat_transfer']
    print(f"    Reynolds number:                {ht['reynolds_number']:.0f}")
    print(f"    Heat transfer coefficient:      {ht['heat_transfer_coeff_W_m2K']:.0f} W/m²/K")

    print("\n  TEMPERATURE DISTRIBUTION")
    print("  " + "-" * 50)
    temps = th['temperatures']
    print(f"    Average fuel centerline:        {temps['avg_fuel_centerline_K']:.1f} K")
    print(f"    Peak fuel centerline:           {temps['peak_fuel_centerline_K']:.1f} K")
    print(f"    Peak cladding surface:          {temps['peak_clad_temp_K']:.1f} K")
    print(f"    Fuel melt margin:               {temps['fuel_melt_margin_K']:.1f} K")

    print("\n  SAFETY MARGINS")
    print("  " + "-" * 50)
    safety = th['safety_margins']
    print(f"    Minimum DNBR:                   {safety['minimum_dnbr']:.2f}")
    print(f"    DNBR limit:                     {safety['dnbr_limit']:.2f}")
    print(f"    DNBR margin:                    {safety['dnbr_margin']:.2f}")

    print("\n  PRESSURE DROP")
    print("  " + "-" * 50)
    dp = th['pressure_drop']
    print(f"    Friction:                       {dp['friction']:.1f} kPa")
    print(f"    Gravity:                        {dp['gravity']:.1f} kPa")
    print(f"    Form losses:                    {dp['form_losses']:.1f} kPa")
    print(f"    Acceleration:                   {dp['acceleration']:.1f} kPa")
    print(f"    TOTAL:                          {dp['total']:.1f} kPa")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PWR Core Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run basic simulation with defaults
  %(prog)s --power 2800 --enrichment 4.5
  %(prog)s --study parametric       # Run parametric study
  %(prog)s --study all              # Run all analyses
        """
    )

    parser.add_argument(
        "--power",
        type=float,
        default=3000.0,
        help="Thermal power in MW (default: 3000)"
    )
    parser.add_argument(
        "--enrichment",
        type=float,
        default=4.0,
        help="U-235 enrichment in weight %% (default: 4.0, range: 3-5)"
    )
    parser.add_argument(
        "--study",
        choices=["parametric", "flux", "criticality", "thermal", "all"],
        help="Run specific study type"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path"
    )

    args = parser.parse_args()

    # Validate enrichment
    if not 3.0 <= args.enrichment <= 5.0:
        print(f"Error: Enrichment must be between 3-5%, got {args.enrichment}%")
        sys.exit(1)

    try:
        if args.study == "parametric":
            run_parametric_study()
        elif args.study == "flux":
            run_flux_calculation()
        elif args.study == "criticality":
            run_criticality_breakdown()
        elif args.study == "thermal":
            run_thermal_analysis()
        elif args.study == "all":
            run_basic_simulation(args.power, args.enrichment)
            run_parametric_study()
            run_flux_calculation()
            run_criticality_breakdown()
            run_thermal_analysis()
        else:
            reactor = run_basic_simulation(args.power, args.enrichment)

            if args.output:
                reactor.to_json(args.output)
                print(f"\nResults exported to: {args.output}")

    except Exception as e:
        print(f"\nError during simulation: {e}")
        raise


if __name__ == "__main__":
    main()
