"""
Neutronics Module for PWR Core Modeling

This module implements neutron physics calculations including:
- Four-factor and six-factor formula for criticality
- Neutron flux distribution
- Two-group diffusion theory
- Reactivity calculations
- Fission product poisoning (Xe-135, Sm-149)

The calculations follow standard reactor physics methodology as
described in Duderstadt & Hamilton and Lamarsh textbooks.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import math
import numpy as np

from .constants import (
    PhysicalConstants,
    NuclearData,
    DELAYED_NEUTRON_DATA,
    FISSION_PRODUCT_DATA,
    TWO_GROUP_STRUCTURE,
)
from .materials import UO2Fuel, LightWaterModerator, FuelAssembly
from .geometry import CoreGeometry


@dataclass
class CriticalityCalculator:
    """
    Calculate reactor criticality using four-factor and six-factor formulas.

    The multiplication factor k determines if the reactor is:
    - Subcritical (k < 1): Chain reaction dies out
    - Critical (k = 1): Steady-state operation
    - Supercritical (k > 1): Power increasing

    Four-factor formula (infinite medium):
        k_inf = η * ε * p * f

    Six-factor formula (finite reactor):
        k_eff = η * ε * p * f * P_F * P_T

    where:
        η = reproduction factor
        ε = fast fission factor
        p = resonance escape probability
        f = thermal utilization factor
        P_F = fast non-leakage probability
        P_T = thermal non-leakage probability
    """

    fuel: UO2Fuel
    moderator: LightWaterModerator
    assembly: FuelAssembly
    geometry: CoreGeometry
    temperature: float = 573.0  # [K] operating temperature
    _nuclear_data: NuclearData = field(default_factory=NuclearData, repr=False)

    def calculate_eta(self) -> float:
        """
        Calculate reproduction factor η.

        η = ν * σ_f / σ_a for U-235

        This represents the average number of neutrons produced
        per neutron absorbed in fuel.

        Returns:
            Reproduction factor η
        """
        # For fuel mixture
        nu = self._nuclear_data.u235["nu"]
        sigma_f_235 = self._nuclear_data.u235["sigma_f"]
        sigma_a_235 = self._nuclear_data.get_u235_sigma_a()
        sigma_a_238 = self._nuclear_data.get_u238_sigma_a()

        # Weight by number densities
        n_235 = self.fuel.n_u235
        n_238 = self.fuel.n_u238

        # η = ν₂₃₅ * Σf_235 / (Σa_235 + Σa_238)
        numerator = nu * n_235 * sigma_f_235
        denominator = n_235 * sigma_a_235 + n_238 * sigma_a_238

        return numerator / denominator

    def calculate_epsilon(self) -> float:
        """
        Calculate fast fission factor ε.

        ε accounts for fissions in U-238 by fast neutrons
        before they slow down to thermal energies.

        For typical PWR lattices: ε ≈ 1.02-1.08

        Using empirical correlation:
        ε = 1 + (1-p) * (ν_fast - 1) * σ_f_238 / σ_a_238 * P_fast

        Returns:
            Fast fission factor ε
        """
        # Simplified model for PWR
        # Based on fuel-to-moderator ratio
        v_f = self.assembly.fuel_volume_fraction
        v_m = self.assembly.moderator_volume_fraction

        # Empirical correlation for PWR
        # ε increases with fuel concentration
        fuel_to_mod_ratio = v_f / v_m

        # Empirical fit for UO2-H2O lattices
        epsilon = 1.0 + 0.29 * fuel_to_mod_ratio

        return min(epsilon, 1.10)  # Physical upper bound

    def calculate_resonance_escape_probability(self) -> float:
        """
        Calculate resonance escape probability p.

        p = exp(-N_238 * I / ξ * Σ_s)

        where I is the effective resonance integral and
        ξΣ_s is the slowing-down power.

        For heterogeneous lattices, use effective resonance
        integral which accounts for self-shielding.

        Returns:
            Resonance escape probability p
        """
        # U-238 resonance integral (effective, with self-shielding)
        # Depends on fuel geometry - use Nordheim correlation
        I_0 = self._nuclear_data.u238["resonance_integral"]  # Infinite dilution

        # Self-shielding factor (Nordheim-like correlation)
        # Depends on surface-to-mass ratio
        fuel_radius = 0.41  # [cm] typical
        fuel_density = self.fuel.density
        S_M = 3.0 / (fuel_radius * fuel_density)  # Surface to mass ratio

        # Effective resonance integral
        # Empirical correlation for UO2 fuel
        I_eff = 4.45 + 26.6 * math.sqrt(S_M)

        # N_238 in moderator volume
        n_238 = self.fuel.n_u238 * self.assembly.fuel_volume_fraction

        # Slowing down power of moderator
        xi_sigma_s = self.moderator.get_slowing_down_power()

        # Volume-weighted
        xi_sigma_s_eff = xi_sigma_s * self.assembly.moderator_volume_fraction

        # Resonance escape probability
        barn_to_cm2 = PhysicalConstants.BARN_TO_CM2
        exponent = -n_238 * I_eff * barn_to_cm2 / xi_sigma_s_eff

        return math.exp(exponent)

    def calculate_thermal_utilization(self) -> float:
        """
        Calculate thermal utilization factor f.

        f = Σ_a_fuel / Σ_a_total

        This is the probability that a thermal neutron is
        absorbed in the fuel (rather than moderator or structure).

        Returns:
            Thermal utilization factor f
        """
        # Macroscopic absorption cross-sections
        sigma_a_fuel = self.fuel.get_macroscopic_absorption_xs()
        sigma_a_mod = self.moderator.get_macroscopic_absorption_xs()
        sigma_a_clad = self.assembly.cladding.get_macroscopic_absorption_xs()

        # Volume fractions
        v_f = self.assembly.fuel_volume_fraction
        v_m = self.assembly.moderator_volume_fraction
        v_c = self.assembly.cladding_volume_fraction

        # Thermal disadvantage factor (flux depression in fuel)
        # For typical PWR: ζ ≈ 0.9-0.95
        zeta = 0.93

        # Homogenized absorption
        sigma_a_total = (
            zeta * v_f * sigma_a_fuel +
            v_m * sigma_a_mod +
            v_c * sigma_a_clad
        )

        return (zeta * v_f * sigma_a_fuel) / sigma_a_total

    def calculate_fast_non_leakage(self, B_sq: Optional[float] = None) -> float:
        """
        Calculate fast non-leakage probability P_F.

        P_F = 1 / (1 + τ * B²)

        where τ is the Fermi age and B² is geometric buckling.

        Args:
            B_sq: Geometric buckling [1/cm²]. If None, calculated from geometry.

        Returns:
            Fast non-leakage probability P_F
        """
        if B_sq is None:
            _, _, B_sq = self.geometry.get_buckling_geometric()

        # Fermi age τ [cm²]
        # For light water at PWR conditions: τ ≈ 27-33 cm²
        tau = self._calculate_fermi_age()

        return 1.0 / (1.0 + tau * B_sq)

    def calculate_thermal_non_leakage(self, B_sq: Optional[float] = None) -> float:
        """
        Calculate thermal non-leakage probability P_T.

        P_T = 1 / (1 + L² * B²)

        where L is the thermal diffusion length.

        Args:
            B_sq: Geometric buckling [1/cm²]. If None, calculated from geometry.

        Returns:
            Thermal non-leakage probability P_T
        """
        if B_sq is None:
            _, _, B_sq = self.geometry.get_buckling_geometric()

        # Diffusion length squared
        L_sq = self.moderator.get_diffusion_length()**2

        # Account for fuel absorption reducing effective L
        # For PWR: L² ≈ 3-6 cm²
        f = self.calculate_thermal_utilization()
        L_sq_eff = L_sq * (1 - f)

        return 1.0 / (1.0 + L_sq_eff * B_sq)

    def _calculate_fermi_age(self) -> float:
        """
        Calculate Fermi age for slowing down.

        τ = ∫(D/ξΣ_s) dE from thermal to fission energy

        For water moderated systems, τ is dominated by hydrogen.

        Returns:
            Fermi age [cm²]
        """
        # For H2O moderator at PWR operating conditions
        # Temperature correction
        T_ratio = self.temperature / 293.0

        # Base value at room temperature
        tau_ref = 27.0  # [cm²] for pure H2O

        # Temperature and density correction
        rho_ratio = self.moderator.density / 1.0
        tau = tau_ref / (rho_ratio**2) * T_ratio

        return tau

    def calculate_k_infinity(self) -> Tuple[float, Dict[str, float]]:
        """
        Calculate infinite multiplication factor k_inf.

        k_inf = η * ε * p * f

        Returns:
            Tuple of (k_inf, dictionary of individual factors)
        """
        eta = self.calculate_eta()
        epsilon = self.calculate_epsilon()
        p = self.calculate_resonance_escape_probability()
        f = self.calculate_thermal_utilization()

        k_inf = eta * epsilon * p * f

        factors = {
            "eta": eta,
            "epsilon": epsilon,
            "p": p,
            "f": f,
            "k_infinity": k_inf,
        }

        return k_inf, factors

    def calculate_k_effective(self) -> Tuple[float, Dict[str, float]]:
        """
        Calculate effective multiplication factor k_eff.

        k_eff = k_inf * P_F * P_T = η * ε * p * f * P_F * P_T

        Returns:
            Tuple of (k_eff, dictionary of all six factors)
        """
        k_inf, factors = self.calculate_k_infinity()

        P_F = self.calculate_fast_non_leakage()
        P_T = self.calculate_thermal_non_leakage()

        k_eff = k_inf * P_F * P_T

        factors.update({
            "P_fast": P_F,
            "P_thermal": P_T,
            "k_effective": k_eff,
        })

        return k_eff, factors

    def calculate_reactivity(self) -> float:
        """
        Calculate reactivity ρ.

        ρ = (k - 1) / k

        Returns:
            Reactivity in Δk/k (dimensionless)
        """
        k_eff, _ = self.calculate_k_effective()
        return (k_eff - 1.0) / k_eff

    def calculate_reactivity_pcm(self) -> float:
        """
        Calculate reactivity in pcm (percent mille).

        1 pcm = 10^-5 Δk/k

        Returns:
            Reactivity in pcm
        """
        return self.calculate_reactivity() * 1e5

    def calculate_reactivity_dollars(self) -> float:
        """
        Calculate reactivity in dollars.

        $1 = β (delayed neutron fraction)

        Returns:
            Reactivity in dollars
        """
        rho = self.calculate_reactivity()
        beta = DELAYED_NEUTRON_DATA["beta_total"]
        return rho / beta


@dataclass
class NeutronicsModel:
    """
    Comprehensive neutronics model for the reactor core.

    This class combines criticality calculations with spatial
    flux distributions and power profiles.
    """

    geometry: CoreGeometry
    fuel: UO2Fuel
    moderator: LightWaterModerator
    assembly: FuelAssembly
    criticality: CriticalityCalculator = field(init=False)
    _nuclear_data: NuclearData = field(default_factory=NuclearData, repr=False)

    def __post_init__(self):
        """Initialize criticality calculator."""
        self.criticality = CriticalityCalculator(
            fuel=self.fuel,
            moderator=self.moderator,
            assembly=self.assembly,
            geometry=self.geometry,
        )

    def calculate_average_flux(self) -> float:
        """
        Calculate core-average neutron flux.

        φ_avg = P / (Σ_f * E_f * V)

        where P is thermal power, Σ_f is macroscopic fission
        cross-section, E_f is energy per fission, and V is
        fuel volume.

        Returns:
            Average thermal neutron flux [n/cm²/s]
        """
        power_watts = self.geometry.thermal_power * 1e6  # Convert MW to W
        energy_per_fission = PhysicalConstants.ENERGY_PER_FISSION_J

        # Macroscopic fission cross-section (volume-weighted)
        sigma_f = (
            self.fuel.get_macroscopic_fission_xs() *
            self.assembly.fuel_volume_fraction
        )

        fuel_volume = self.geometry.total_fuel_volume

        # φ = P / (Σ_f * E_f * V)
        flux = power_watts / (sigma_f * energy_per_fission * fuel_volume)

        return flux

    def calculate_peak_flux(self) -> float:
        """
        Calculate peak neutron flux.

        φ_peak = φ_avg * F_total

        where F_total is the total peaking factor.

        Returns:
            Peak thermal neutron flux [n/cm²/s]
        """
        avg_flux = self.calculate_average_flux()
        peaking = self.geometry.get_peaking_factors()
        return avg_flux * peaking["total"]

    def calculate_flux_distribution(
        self,
        r_points: int = 50,
        z_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate 2D flux distribution φ(r,z).

        Uses fundamental mode approximation:
        φ(r,z) = φ_0 * J_0(2.405*r/R) * cos(πz/H)

        Args:
            r_points: Number of radial points
            z_points: Number of axial points

        Returns:
            Tuple of (r_array, z_array, flux_2d_array)
        """
        from scipy.special import j0

        R = self.geometry.equivalent_radius
        H = self.geometry.core_height

        # Create mesh
        r = np.linspace(0, R, r_points)
        z = np.linspace(-H/2, H/2, z_points)
        R_mesh, Z_mesh = np.meshgrid(r, z)

        # Extrapolated dimensions
        R_ex = R + 7.0  # Reflector savings
        H_ex = H + 14.0

        # Flux distribution (fundamental mode)
        phi_0 = self.calculate_peak_flux()
        flux = phi_0 * j0(2.405 * R_mesh / R_ex) * np.cos(np.pi * Z_mesh / H_ex)

        # Set negative values to zero (outside extrapolated boundary)
        flux = np.maximum(flux, 0)

        return r, z, flux

    def calculate_power_distribution(
        self,
        r_points: int = 50,
        z_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate 2D power distribution P(r,z).

        Power is proportional to Σ_f * φ

        Args:
            r_points: Number of radial points
            z_points: Number of axial points

        Returns:
            Tuple of (r_array, z_array, power_2d_array [W/cm³])
        """
        r, z, flux = self.calculate_flux_distribution(r_points, z_points)

        sigma_f = self.fuel.get_macroscopic_fission_xs()
        energy_per_fission = PhysicalConstants.ENERGY_PER_FISSION_J

        # Power density = Σ_f * φ * E_f
        power = (
            sigma_f *
            flux *
            energy_per_fission *
            self.assembly.fuel_volume_fraction
        )

        return r, z, power

    def calculate_xenon_worth(self, flux: Optional[float] = None) -> float:
        """
        Calculate equilibrium Xe-135 reactivity worth.

        Xe-135 is a strong neutron absorber produced by fission.
        At equilibrium:
        ρ_Xe = -σ_Xe * N_Xe / Σ_a

        Args:
            flux: Neutron flux [n/cm²/s]. If None, use average flux.

        Returns:
            Xenon reactivity worth [Δk/k]
        """
        if flux is None:
            flux = self.calculate_average_flux()

        xe_data = FISSION_PRODUCT_DATA["Xe-135"]
        sigma_xe = xe_data["sigma_a"]
        lambda_xe = xe_data["decay_constant"]
        gamma_xe = xe_data["yield"]

        # I-135 data (Xe precursor)
        i_data = FISSION_PRODUCT_DATA["I-135"]
        lambda_i = i_data["decay_constant"]
        gamma_i = i_data["yield"]

        # Equilibrium Xe concentration
        sigma_f = self.fuel.get_macroscopic_fission_xs()
        sigma_a_xe = sigma_xe * PhysicalConstants.BARN_TO_CM2

        # N_Xe = Σ_f * φ * (γ_Xe + γ_I) / (λ_Xe + σ_Xe * φ)
        N_xe = (
            sigma_f * flux * (gamma_xe + gamma_i) /
            (lambda_xe + sigma_a_xe * flux)
        )

        # Reactivity worth
        sigma_a_total = self.assembly.get_homogenized_sigma_a()
        rho_xe = -N_xe * sigma_a_xe / sigma_a_total

        return rho_xe

    def calculate_samarium_worth(self) -> float:
        """
        Calculate equilibrium Sm-149 reactivity worth.

        Sm-149 is stable and builds up to equilibrium.

        Returns:
            Samarium reactivity worth [Δk/k]
        """
        sm_data = FISSION_PRODUCT_DATA["Sm-149"]
        sigma_sm = sm_data["sigma_a"]
        gamma_sm = sm_data["yield"]

        flux = self.calculate_average_flux()
        sigma_f = self.fuel.get_macroscopic_fission_xs()
        sigma_a_sm = sigma_sm * PhysicalConstants.BARN_TO_CM2

        # At equilibrium (Sm-149 is stable):
        # N_Sm = γ_Sm * Σ_f / σ_Sm (independent of flux)
        N_sm = gamma_sm * sigma_f / sigma_a_sm

        # Reactivity worth
        sigma_a_total = self.assembly.get_homogenized_sigma_a()
        rho_sm = -N_sm * sigma_a_sm / sigma_a_total

        return rho_sm

    def calculate_neutron_lifetime(self) -> float:
        """
        Calculate prompt neutron lifetime.

        l = 1 / (v * Σ_a)

        Returns:
            Prompt neutron lifetime [s]
        """
        v_thermal = PhysicalConstants.THERMAL_NEUTRON_VELOCITY
        sigma_a = self.assembly.get_homogenized_sigma_a()

        # Account for temperature (faster neutrons at higher T)
        T_ratio = math.sqrt(self.moderator.temperature / 293.0)
        v_actual = v_thermal * T_ratio

        return 1.0 / (v_actual * sigma_a * 100)  # Convert to seconds

    def calculate_migration_length(self) -> float:
        """
        Calculate migration length M.

        M² = L² + τ

        Returns:
            Migration length [cm]
        """
        L_sq = self.moderator.get_diffusion_length()**2
        tau = self.criticality._calculate_fermi_age()
        return math.sqrt(L_sq + tau)


@dataclass
class TwoGroupDiffusion:
    """
    Two-group neutron diffusion model.

    Solves the two-group diffusion equations:
    -D₁∇²φ₁ + Σ_r1 φ₁ = χ₁/k (νΣ_f1 φ₁ + νΣ_f2 φ₂)
    -D₂∇²φ₂ + Σ_a2 φ₂ = χ₂/k (νΣ_f1 φ₁ + νΣ_f2 φ₂) + Σ_s12 φ₁
    """

    geometry: CoreGeometry
    fuel: UO2Fuel
    moderator: LightWaterModerator
    assembly: FuelAssembly
    _constants: PhysicalConstants = field(
        default_factory=PhysicalConstants, repr=False
    )
    _nuclear_data: NuclearData = field(default_factory=NuclearData, repr=False)

    def get_two_group_constants(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate two-group cross-sections and diffusion coefficients.

        Returns:
            Dictionary with 'fast' and 'thermal' group constants
        """
        v_f = self.assembly.fuel_volume_fraction
        v_m = self.assembly.moderator_volume_fraction
        barn_to_cm2 = self._constants.BARN_TO_CM2

        # Fast group constants (simplified)
        D1 = 1.5  # [cm] approximate for fast group in water
        sigma_r1 = 0.025  # [1/cm] removal cross-section (slowing down)
        nu_sigma_f1 = (
            self.fuel.n_u238 *
            self._nuclear_data.u238["sigma_f"] *
            2.5 *  # Fast nu
            barn_to_cm2 *
            v_f
        )

        # Thermal group constants
        D2 = self.moderator.get_diffusion_coefficient()
        sigma_a2 = self.assembly.get_homogenized_sigma_a()
        nu_sigma_f2 = self.assembly.get_homogenized_nu_sigma_f()

        return {
            "fast": {
                "D": D1,
                "sigma_removal": sigma_r1,
                "nu_sigma_f": nu_sigma_f1,
                "chi": TWO_GROUP_STRUCTURE["fast"]["chi"],
            },
            "thermal": {
                "D": D2,
                "sigma_a": sigma_a2,
                "nu_sigma_f": nu_sigma_f2,
                "chi": TWO_GROUP_STRUCTURE["thermal"]["chi"],
            }
        }

    def calculate_k_two_group(self) -> float:
        """
        Calculate k_eff using two-group theory.

        For a bare reactor:
        k = (νΣ_f1 + νΣ_f2 * Σ_r1/(Σ_a2 + D2*B²)) / (Σ_r1 + D1*B²)

        Returns:
            k_effective from two-group calculation
        """
        constants = self.get_two_group_constants()
        _, _, B_sq = self.geometry.get_buckling_geometric()

        D1 = constants["fast"]["D"]
        sigma_r1 = constants["fast"]["sigma_removal"]
        nu_sigma_f1 = constants["fast"]["nu_sigma_f"]

        D2 = constants["thermal"]["D"]
        sigma_a2 = constants["thermal"]["sigma_a"]
        nu_sigma_f2 = constants["thermal"]["nu_sigma_f"]

        # Thermal group denominator
        thermal_denom = sigma_a2 + D2 * B_sq

        # Fast group denominator
        fast_denom = sigma_r1 + D1 * B_sq

        # k calculation
        k = (nu_sigma_f1 + nu_sigma_f2 * sigma_r1 / thermal_denom) / fast_denom

        return k

    def calculate_flux_ratio(self) -> float:
        """
        Calculate thermal-to-fast flux ratio φ₂/φ₁.

        Returns:
            Flux ratio (thermal/fast)
        """
        constants = self.get_two_group_constants()
        _, _, B_sq = self.geometry.get_buckling_geometric()

        D2 = constants["thermal"]["D"]
        sigma_a2 = constants["thermal"]["sigma_a"]
        sigma_r1 = constants["fast"]["sigma_removal"]

        # φ₂/φ₁ = Σ_r1 / (Σ_a2 + D2*B²)
        ratio = sigma_r1 / (sigma_a2 + D2 * B_sq)

        return ratio
