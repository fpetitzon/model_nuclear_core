"""
Main PWR Core Reactor Model

This module provides the top-level reactor model that integrates
all physics components: neutronics, thermal-hydraulics, and materials.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any
import math
import json
from datetime import datetime

from .constants import PhysicalConstants, NuclearData, DELAYED_NEUTRON_DATA
from .materials import UO2Fuel, LightWaterModerator, FuelAssembly, ZircaloyCladding
from .geometry import CoreGeometry, FuelPin, create_standard_pwr_geometry
from .neutronics import NeutronicsModel, CriticalityCalculator, TwoGroupDiffusion
from .thermal import ThermalHydraulics, CoolantProperties


@dataclass
class PWRCore:
    """
    Complete Pressurized Water Reactor Core Model.

    This class integrates all physics models to provide a comprehensive
    simulation of a 3000 MW PWR core.

    Attributes:
        thermal_power: Thermal power output [MW]
        enrichment: U-235 enrichment [weight %]
        coolant_inlet_temp: Coolant inlet temperature [K]
        coolant_outlet_temp: Coolant outlet temperature [K]
        system_pressure: System pressure [MPa]
    """

    thermal_power: float = 3000.0  # [MW]
    enrichment: float = 4.0  # [weight %]
    coolant_inlet_temp: float = 565.0  # [K]
    coolant_outlet_temp: float = 598.0  # [K]
    system_pressure: float = 15.5  # [MPa]
    num_assemblies: int = 193
    pins_per_assembly: int = 264

    # Computed components (initialized in __post_init__)
    fuel: UO2Fuel = field(init=False)
    moderator: LightWaterModerator = field(init=False)
    assembly: FuelAssembly = field(init=False)
    geometry: CoreGeometry = field(init=False)
    neutronics: NeutronicsModel = field(init=False)
    thermal: ThermalHydraulics = field(init=False)
    two_group: TwoGroupDiffusion = field(init=False)

    def __post_init__(self):
        """Initialize all physics components."""
        # Validate enrichment
        if not 3.0 <= self.enrichment <= 5.0:
            raise ValueError(
                f"Enrichment must be between 3-5%, got {self.enrichment}%"
            )

        # Initialize materials
        avg_temp = (self.coolant_inlet_temp + self.coolant_outlet_temp) / 2

        self.fuel = UO2Fuel(
            enrichment=self.enrichment,
            temperature=900.0,  # Average fuel temperature
        )

        self.moderator = LightWaterModerator(
            temperature=avg_temp,
            pressure=self.system_pressure,
        )

        self.assembly = FuelAssembly(
            fuel=self.fuel,
            moderator=self.moderator,
            fuel_pin_count=self.pins_per_assembly,
        )

        # Initialize geometry
        self.geometry = CoreGeometry(
            thermal_power=self.thermal_power,
            num_assemblies=self.num_assemblies,
            pins_per_assembly=self.pins_per_assembly,
        )

        # Initialize physics models
        self.neutronics = NeutronicsModel(
            geometry=self.geometry,
            fuel=self.fuel,
            moderator=self.moderator,
            assembly=self.assembly,
        )

        self.thermal = ThermalHydraulics(
            geometry=self.geometry,
            inlet_temperature=self.coolant_inlet_temp,
            outlet_temperature=self.coolant_outlet_temp,
            system_pressure=self.system_pressure,
        )

        self.two_group = TwoGroupDiffusion(
            geometry=self.geometry,
            fuel=self.fuel,
            moderator=self.moderator,
            assembly=self.assembly,
        )

    def calculate_criticality(self) -> Dict[str, Any]:
        """
        Calculate complete criticality analysis.

        Returns dictionary with:
        - Four-factor formula components
        - Six-factor formula (k-effective)
        - Reactivity in various units
        - Two-group k-effective
        """
        k_eff, factors = self.neutronics.criticality.calculate_k_effective()

        # Two-group calculation
        k_two_group = self.two_group.calculate_k_two_group()

        # Reactivity
        rho = (k_eff - 1.0) / k_eff
        rho_pcm = rho * 1e5
        rho_dollars = rho / DELAYED_NEUTRON_DATA["beta_total"]

        # Geometric buckling
        b_r, b_z, b_total = self.geometry.get_buckling_geometric()

        return {
            "four_factor": {
                "eta": factors["eta"],
                "epsilon": factors["epsilon"],
                "p": factors["p"],
                "f": factors["f"],
                "k_infinity": factors["k_infinity"],
            },
            "six_factor": {
                "P_fast": factors["P_fast"],
                "P_thermal": factors["P_thermal"],
                "k_effective": k_eff,
            },
            "two_group_k_effective": k_two_group,
            "reactivity": {
                "delta_k_over_k": rho,
                "pcm": rho_pcm,
                "dollars": rho_dollars,
            },
            "buckling": {
                "radial_cm2": b_r,
                "axial_cm2": b_z,
                "total_cm2": b_total,
            },
        }

    def calculate_neutron_flux(self) -> Dict[str, Any]:
        """
        Calculate neutron flux characteristics.

        Returns dictionary with flux values and distributions.
        """
        avg_flux = self.neutronics.calculate_average_flux()
        peak_flux = self.neutronics.calculate_peak_flux()
        flux_ratio = self.two_group.calculate_flux_ratio()

        # Migration length
        M = self.neutronics.calculate_migration_length()

        # Neutron lifetime
        lifetime = self.neutronics.calculate_neutron_lifetime()

        return {
            "average_flux_n_cm2_s": avg_flux,
            "peak_flux_n_cm2_s": peak_flux,
            "thermal_to_fast_ratio": flux_ratio,
            "migration_length_cm": M,
            "prompt_neutron_lifetime_s": lifetime,
            "peaking_factors": self.geometry.get_peaking_factors(),
        }

    def calculate_fission_product_poisoning(self) -> Dict[str, Any]:
        """
        Calculate fission product poisoning effects.

        Returns xenon and samarium worth at equilibrium.
        """
        xe_worth = self.neutronics.calculate_xenon_worth()
        sm_worth = self.neutronics.calculate_samarium_worth()

        return {
            "xenon_135": {
                "reactivity_delta_k": xe_worth,
                "reactivity_pcm": xe_worth * 1e5,
                "reactivity_dollars": xe_worth / DELAYED_NEUTRON_DATA["beta_total"],
            },
            "samarium_149": {
                "reactivity_delta_k": sm_worth,
                "reactivity_pcm": sm_worth * 1e5,
                "reactivity_dollars": sm_worth / DELAYED_NEUTRON_DATA["beta_total"],
            },
            "total_poison_worth_pcm": (xe_worth + sm_worth) * 1e5,
        }

    def calculate_thermal_hydraulics(self) -> Dict[str, Any]:
        """
        Calculate thermal-hydraulic conditions.

        Returns comprehensive thermal-hydraulic data.
        """
        summary = self.thermal.get_thermal_summary()
        pressure_drop = self.thermal.calculate_pressure_drop()
        hot_channel = self.thermal.calculate_hot_channel_temperatures()

        return {
            "bulk_conditions": {
                "inlet_temp_K": self.coolant_inlet_temp,
                "outlet_temp_K": self.coolant_outlet_temp,
                "temp_rise_K": self.coolant_outlet_temp - self.coolant_inlet_temp,
                "mass_flow_kg_s": summary["mass_flow_rate_kg_s"],
                "coolant_velocity_m_s": summary["coolant_velocity_m_s"],
            },
            "heat_transfer": {
                "reynolds_number": summary["reynolds_number"],
                "heat_transfer_coeff_W_m2K": summary["heat_transfer_coeff_W_m2K"],
            },
            "temperatures": {
                "avg_fuel_centerline_K": summary["avg_fuel_centerline_K"],
                "peak_fuel_centerline_K": summary["peak_fuel_centerline_K"],
                "peak_clad_temp_K": hot_channel["peak_clad_temp"],
                "fuel_melt_margin_K": summary["fuel_melt_margin_K"],
            },
            "safety_margins": {
                "minimum_dnbr": summary["minimum_dnbr"],
                "dnbr_limit": 1.3,
                "dnbr_margin": summary["minimum_dnbr"] - 1.3,
            },
            "pressure_drop": pressure_drop,
        }

    def calculate_core_inventory(self) -> Dict[str, Any]:
        """
        Calculate core material inventory.

        Returns fuel loading, uranium inventory, etc.
        """
        total_fuel_mass = self.geometry.total_fuel_mass  # [kg] UO2

        # Uranium mass (excluding oxygen)
        u_fraction = 238.0 / 270.0  # Approximate U/UO2 mass ratio
        u_mass = total_fuel_mass * u_fraction

        # U-235 mass
        u235_mass = u_mass * (self.enrichment / 100.0)

        return {
            "total_fuel_pins": self.geometry.total_fuel_pins,
            "total_fuel_volume_cm3": self.geometry.total_fuel_volume,
            "uo2_mass_kg": total_fuel_mass,
            "uranium_mass_kg": u_mass,
            "u235_mass_kg": u235_mass,
            "u238_mass_kg": u_mass - u235_mass,
            "specific_power_MW_tHM": self.geometry.specific_power,
            "power_density_MW_m3": self.geometry.power_density,
        }

    def get_geometry_summary(self) -> Dict[str, Any]:
        """
        Get core geometry summary.
        """
        return {
            "core": {
                "thermal_power_MW": self.thermal_power,
                "num_assemblies": self.num_assemblies,
                "pins_per_assembly": self.pins_per_assembly,
                "total_fuel_pins": self.geometry.total_fuel_pins,
                "equivalent_radius_cm": self.geometry.equivalent_radius,
                "active_height_cm": self.geometry.core_height,
                "core_volume_m3": self.geometry.core_volume * 1e-6,
            },
            "fuel_pin": {
                "fuel_radius_cm": self.geometry.fuel_pin.fuel_radius,
                "clad_outer_radius_cm": self.geometry.fuel_pin.clad_outer_radius,
                "clad_thickness_cm": self.geometry.fuel_pin.clad_thickness,
                "active_length_cm": self.geometry.fuel_pin.active_length,
            },
            "assembly": {
                "array_size": "17x17",
                "pitch_cm": self.assembly.pitch,
                "assembly_pitch_cm": self.assembly.assembly_pitch,
                "fuel_volume_fraction": self.assembly.fuel_volume_fraction,
                "moderator_volume_fraction": self.assembly.moderator_volume_fraction,
                "H_to_U_ratio": self.assembly.get_moderation_ratio(),
            },
        }

    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete reactor core analysis.

        Returns comprehensive dictionary with all calculated parameters.
        """
        analysis = {
            "metadata": {
                "model": "PWR Core Model v1.0",
                "timestamp": datetime.now().isoformat(),
                "thermal_power_MW": self.thermal_power,
                "enrichment_percent": self.enrichment,
            },
            "geometry": self.get_geometry_summary(),
            "criticality": self.calculate_criticality(),
            "neutron_flux": self.calculate_neutron_flux(),
            "fission_product_poisoning": self.calculate_fission_product_poisoning(),
            "thermal_hydraulics": self.calculate_thermal_hydraulics(),
            "core_inventory": self.calculate_core_inventory(),
        }

        return analysis

    def print_summary(self):
        """Print formatted summary of reactor analysis."""
        analysis = self.run_full_analysis()

        print("=" * 70)
        print("           PWR CORE ANALYSIS SUMMARY")
        print("=" * 70)

        print(f"\n{'CORE PARAMETERS':^70}")
        print("-" * 70)
        print(f"  Thermal Power:          {self.thermal_power:>10.1f} MW")
        print(f"  U-235 Enrichment:       {self.enrichment:>10.2f} %")
        print(f"  Number of Assemblies:   {self.num_assemblies:>10d}")
        print(f"  Total Fuel Pins:        {self.geometry.total_fuel_pins:>10d}")

        crit = analysis["criticality"]
        print(f"\n{'CRITICALITY ANALYSIS':^70}")
        print("-" * 70)
        print("  Four-Factor Formula:")
        print(f"    η (reproduction):     {crit['four_factor']['eta']:>10.4f}")
        print(f"    ε (fast fission):     {crit['four_factor']['epsilon']:>10.4f}")
        print(f"    p (resonance escape): {crit['four_factor']['p']:>10.4f}")
        print(f"    f (thermal util.):    {crit['four_factor']['f']:>10.4f}")
        print(f"    k_∞:                  {crit['four_factor']['k_infinity']:>10.4f}")
        print("  Six-Factor Formula:")
        print(f"    P_F (fast non-leak):  {crit['six_factor']['P_fast']:>10.4f}")
        print(f"    P_T (therm non-leak): {crit['six_factor']['P_thermal']:>10.4f}")
        print(f"    k_eff:                {crit['six_factor']['k_effective']:>10.4f}")
        print(f"  Reactivity:             {crit['reactivity']['pcm']:>10.1f} pcm")

        flux = analysis["neutron_flux"]
        print(f"\n{'NEUTRON FLUX':^70}")
        print("-" * 70)
        print(f"  Average Flux:           {flux['average_flux_n_cm2_s']:>10.3e} n/cm²/s")
        print(f"  Peak Flux:              {flux['peak_flux_n_cm2_s']:>10.3e} n/cm²/s")
        print(f"  Migration Length:       {flux['migration_length_cm']:>10.2f} cm")
        print(f"  Neutron Lifetime:       {flux['prompt_neutron_lifetime_s']:>10.2e} s")

        th = analysis["thermal_hydraulics"]
        print(f"\n{'THERMAL-HYDRAULICS':^70}")
        print("-" * 70)
        print(f"  Coolant Inlet Temp:     {th['bulk_conditions']['inlet_temp_K']:>10.1f} K")
        print(f"  Coolant Outlet Temp:    {th['bulk_conditions']['outlet_temp_K']:>10.1f} K")
        print(f"  Mass Flow Rate:         {th['bulk_conditions']['mass_flow_kg_s']:>10.1f} kg/s")
        print(f"  Peak Fuel Centerline:   {th['temperatures']['peak_fuel_centerline_K']:>10.1f} K")
        print(f"  Fuel Melt Margin:       {th['temperatures']['fuel_melt_margin_K']:>10.1f} K")
        print(f"  Minimum DNBR:           {th['safety_margins']['minimum_dnbr']:>10.2f}")
        print(f"  Core Pressure Drop:     {th['pressure_drop']['total']:>10.1f} kPa")

        poison = analysis["fission_product_poisoning"]
        print(f"\n{'FISSION PRODUCT POISONING':^70}")
        print("-" * 70)
        print(f"  Xe-135 Worth:           {poison['xenon_135']['reactivity_pcm']:>10.1f} pcm")
        print(f"  Sm-149 Worth:           {poison['samarium_149']['reactivity_pcm']:>10.1f} pcm")

        inv = analysis["core_inventory"]
        print(f"\n{'CORE INVENTORY':^70}")
        print("-" * 70)
        print(f"  UO2 Mass:               {inv['uo2_mass_kg']:>10.1f} kg")
        print(f"  U-235 Mass:             {inv['u235_mass_kg']:>10.1f} kg")
        print(f"  Specific Power:         {inv['specific_power_MW_tHM']:>10.1f} MW/tHM")

        print("\n" + "=" * 70)

    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Export analysis results to JSON.

        Args:
            filepath: Optional file path to save JSON

        Returns:
            JSON string
        """
        analysis = self.run_full_analysis()
        json_str = json.dumps(analysis, indent=2, default=str)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str


def create_pwr_core(
    power_mw: float = 3000.0,
    enrichment: float = 4.0,
    **kwargs
) -> PWRCore:
    """
    Factory function to create a PWR core model.

    Args:
        power_mw: Thermal power in MW
        enrichment: U-235 enrichment in weight percent (3-5%)
        **kwargs: Additional parameters passed to PWRCore

    Returns:
        Configured PWRCore instance
    """
    return PWRCore(
        thermal_power=power_mw,
        enrichment=enrichment,
        **kwargs
    )
