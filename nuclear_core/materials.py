"""
Material Properties for PWR Core Modeling

This module defines the fuel (UO2), moderator (light water), and
structural materials (Zircaloy cladding) used in a PWR core.
"""

from dataclasses import dataclass, field
from typing import Optional
import math

from .constants import PhysicalConstants, NuclearData


@dataclass
class UO2Fuel:
    """
    Uranium Dioxide (UO2) fuel material properties.

    UO2 is the standard fuel for PWRs. This class models the fuel
    with variable enrichment between 3-5% U-235.

    Attributes:
        enrichment: U-235 enrichment (weight percent), 3.0-5.0%
        temperature: Fuel temperature [K]
        theoretical_density: Theoretical density of UO2 [g/cm³]
        density_fraction: Fraction of theoretical density (typically 0.95)
    """

    enrichment: float = 4.0  # weight percent U-235
    temperature: float = 900.0  # [K] average fuel temperature
    theoretical_density: float = 10.97  # [g/cm³]
    density_fraction: float = 0.95  # 95% of theoretical density

    def __post_init__(self):
        """Validate enrichment is within PWR range."""
        if not 3.0 <= self.enrichment <= 5.0:
            raise ValueError(
                f"Enrichment must be between 3-5%, got {self.enrichment}%"
            )

        self._constants = PhysicalConstants()
        self._nuclear_data = NuclearData()

    @property
    def density(self) -> float:
        """Actual fuel density [g/cm³]."""
        return self.theoretical_density * self.density_fraction

    @property
    def molecular_weight(self) -> float:
        """
        Calculate molecular weight of UO2 considering enrichment.

        Returns:
            Molecular weight [g/mol]
        """
        # Weight fractions
        w_u235 = self.enrichment / 100.0
        w_u238 = 1.0 - w_u235

        # Average uranium atomic mass
        avg_u_mass = (
            w_u235 * self._nuclear_data.u235["atomic_mass"] +
            w_u238 * self._nuclear_data.u238["atomic_mass"]
        )

        # UO2 molecular weight (U + 2*O)
        return avg_u_mass + 2 * self._nuclear_data.o16["atomic_mass"]

    @property
    def uranium_density(self) -> float:
        """
        Calculate uranium atom density in fuel.

        Returns:
            Uranium number density [atoms/cm³]
        """
        return (
            self.density *
            self._constants.AVOGADRO /
            self.molecular_weight
        )

    @property
    def n_u235(self) -> float:
        """
        U-235 number density [atoms/cm³].

        Returns:
            Number density of U-235 atoms
        """
        return self.uranium_density * (self.enrichment / 100.0)

    @property
    def n_u238(self) -> float:
        """
        U-238 number density [atoms/cm³].

        Returns:
            Number density of U-238 atoms
        """
        return self.uranium_density * (1.0 - self.enrichment / 100.0)

    @property
    def n_oxygen(self) -> float:
        """
        Oxygen number density [atoms/cm³].

        Returns:
            Number density of oxygen atoms (2 per UO2 molecule)
        """
        return 2.0 * self.uranium_density

    def get_macroscopic_fission_xs(self) -> float:
        """
        Calculate macroscopic fission cross-section.

        Σ_f = N_235 * σ_f_235 + N_238 * σ_f_238

        Returns:
            Macroscopic fission cross-section [1/cm]
        """
        barn_to_cm2 = self._constants.BARN_TO_CM2

        sigma_f = (
            self.n_u235 * self._nuclear_data.u235["sigma_f"] * barn_to_cm2 +
            self.n_u238 * self._nuclear_data.u238["sigma_f"] * barn_to_cm2
        )
        return sigma_f

    def get_macroscopic_absorption_xs(self) -> float:
        """
        Calculate macroscopic absorption cross-section.

        Σ_a = Σ_a_235 + Σ_a_238 + Σ_a_O

        Returns:
            Macroscopic absorption cross-section [1/cm]
        """
        barn_to_cm2 = self._constants.BARN_TO_CM2

        sigma_a = (
            self.n_u235 * self._nuclear_data.get_u235_sigma_a() * barn_to_cm2 +
            self.n_u238 * self._nuclear_data.get_u238_sigma_a() * barn_to_cm2 +
            self.n_oxygen * self._nuclear_data.o16["sigma_c"] * barn_to_cm2
        )
        return sigma_a

    def get_nu_sigma_f(self) -> float:
        """
        Calculate ν * Σ_f (neutron production cross-section).

        Returns:
            νΣ_f [1/cm]
        """
        barn_to_cm2 = self._constants.BARN_TO_CM2

        nu_sigma_f = (
            self.n_u235 *
            self._nuclear_data.u235["nu"] *
            self._nuclear_data.u235["sigma_f"] *
            barn_to_cm2
        )
        return nu_sigma_f


@dataclass
class LightWaterModerator:
    """
    Light water (H2O) moderator and coolant properties.

    In a PWR, water serves as both moderator (slowing neutrons)
    and coolant (removing heat from fuel).

    Attributes:
        temperature: Water temperature [K]
        pressure: System pressure [MPa]
    """

    temperature: float = 573.0  # [K] (~300°C)
    pressure: float = 15.5  # [MPa] (typical PWR operating pressure)

    def __post_init__(self):
        self._constants = PhysicalConstants()
        self._nuclear_data = NuclearData()

    @property
    def density(self) -> float:
        """
        Calculate water density at operating conditions.

        Uses simplified correlation for subcooled water density.

        Returns:
            Water density [g/cm³]
        """
        # Simplified correlation for PWR conditions
        # More accurate correlations (IAPWS-IF97) should be used for detailed analysis
        rho_ref = 1.0  # [g/cm³] at reference conditions
        T_ref = 293.15  # [K]

        # Thermal expansion coefficient approximation
        beta = 4.5e-4  # [1/K] approximate for water

        # Pressure correction (compressibility)
        kappa = 4.5e-4  # [1/MPa] approximate

        density = rho_ref * (
            1.0 - beta * (self.temperature - T_ref) +
            kappa * (self.pressure - 0.101325)
        )

        return max(density, 0.5)  # Ensure physical bounds

    @property
    def n_hydrogen(self) -> float:
        """
        Hydrogen number density [atoms/cm³].

        Returns:
            Number density of hydrogen atoms
        """
        n_water = (
            self.density *
            self._constants.AVOGADRO /
            self._nuclear_data.h2o["molecular_mass"]
        )
        return 2.0 * n_water  # 2 H atoms per water molecule

    @property
    def n_oxygen(self) -> float:
        """
        Oxygen number density [atoms/cm³].

        Returns:
            Number density of oxygen atoms
        """
        n_water = (
            self.density *
            self._constants.AVOGADRO /
            self._nuclear_data.h2o["molecular_mass"]
        )
        return n_water  # 1 O atom per water molecule

    def get_macroscopic_absorption_xs(self) -> float:
        """
        Calculate macroscopic absorption cross-section.

        Returns:
            Σ_a [1/cm]
        """
        barn_to_cm2 = self._constants.BARN_TO_CM2

        sigma_a = (
            self.n_hydrogen * self._nuclear_data.h1["sigma_c"] * barn_to_cm2 +
            self.n_oxygen * self._nuclear_data.o16["sigma_c"] * barn_to_cm2
        )
        return sigma_a

    def get_macroscopic_scattering_xs(self) -> float:
        """
        Calculate macroscopic scattering cross-section.

        Returns:
            Σ_s [1/cm]
        """
        barn_to_cm2 = self._constants.BARN_TO_CM2

        sigma_s = (
            self.n_hydrogen * self._nuclear_data.h1["sigma_s"] * barn_to_cm2 +
            self.n_oxygen * self._nuclear_data.o16["sigma_s"] * barn_to_cm2
        )
        return sigma_s

    def get_slowing_down_power(self) -> float:
        """
        Calculate slowing-down power ξΣ_s.

        The slowing-down power represents the effectiveness
        of the moderator in slowing neutrons.

        Returns:
            ξΣ_s [1/cm]
        """
        barn_to_cm2 = self._constants.BARN_TO_CM2

        # Hydrogen dominates slowing down (ξ = 1)
        xi_h = 1.0
        # Oxygen contribution (ξ ≈ 0.12)
        xi_o = 0.12

        slowing_down = (
            xi_h * self.n_hydrogen * self._nuclear_data.h1["sigma_s"] * barn_to_cm2 +
            xi_o * self.n_oxygen * self._nuclear_data.o16["sigma_s"] * barn_to_cm2
        )
        return slowing_down

    def get_moderating_ratio(self) -> float:
        """
        Calculate moderating ratio ξΣ_s/Σ_a.

        The moderating ratio indicates how effectively a material
        slows neutrons relative to absorbing them. Higher is better.

        Returns:
            Moderating ratio (dimensionless)
        """
        return self.get_slowing_down_power() / self.get_macroscopic_absorption_xs()

    def get_diffusion_coefficient(self) -> float:
        """
        Calculate thermal neutron diffusion coefficient.

        D = 1 / (3 * Σ_tr) where Σ_tr ≈ Σ_s (for light elements)

        Returns:
            Diffusion coefficient [cm]
        """
        sigma_tr = self.get_macroscopic_scattering_xs()
        return 1.0 / (3.0 * sigma_tr)

    def get_diffusion_length(self) -> float:
        """
        Calculate thermal diffusion length.

        L = sqrt(D / Σ_a)

        Returns:
            Diffusion length [cm]
        """
        D = self.get_diffusion_coefficient()
        sigma_a = self.get_macroscopic_absorption_xs()
        return math.sqrt(D / sigma_a)


@dataclass
class ZircaloyCladding:
    """
    Zircaloy-4 cladding material properties.

    Zircaloy is used as cladding due to its low neutron absorption
    cross-section and good mechanical properties.

    Attributes:
        outer_radius: Outer radius of cladding [cm]
        thickness: Cladding wall thickness [cm]
        temperature: Cladding temperature [K]
    """

    outer_radius: float = 0.475  # [cm]
    thickness: float = 0.057  # [cm]
    temperature: float = 620.0  # [K]
    density: float = 6.55  # [g/cm³]

    def __post_init__(self):
        self._constants = PhysicalConstants()
        self._nuclear_data = NuclearData()

    @property
    def inner_radius(self) -> float:
        """Inner radius of cladding [cm]."""
        return self.outer_radius - self.thickness

    @property
    def n_zr(self) -> float:
        """Zirconium number density [atoms/cm³]."""
        return (
            self.density *
            self._constants.AVOGADRO /
            self._nuclear_data.zr["atomic_mass"]
        )

    def get_macroscopic_absorption_xs(self) -> float:
        """
        Calculate macroscopic absorption cross-section.

        Returns:
            Σ_a [1/cm]
        """
        barn_to_cm2 = self._constants.BARN_TO_CM2
        return self.n_zr * self._nuclear_data.zr["sigma_c"] * barn_to_cm2


@dataclass
class FuelAssembly:
    """
    PWR fuel assembly containing multiple fuel pins.

    A typical PWR fuel assembly contains a 17x17 array of positions,
    with some positions occupied by control rod guide tubes.

    Attributes:
        array_size: Number of pins in each direction (17 for 17x17)
        fuel_pin_count: Number of fuel pins (264 for typical 17x17)
        guide_tube_count: Number of guide tube positions (25 typical)
        pitch: Pin-to-pin pitch [cm]
        fuel: Fuel material
        moderator: Moderator material
    """

    array_size: int = 17
    fuel_pin_count: int = 264
    guide_tube_count: int = 25
    pitch: float = 1.26  # [cm]
    fuel: UO2Fuel = field(default_factory=UO2Fuel)
    moderator: LightWaterModerator = field(default_factory=LightWaterModerator)
    cladding: ZircaloyCladding = field(default_factory=ZircaloyCladding)

    @property
    def assembly_pitch(self) -> float:
        """Assembly width [cm]."""
        return self.array_size * self.pitch

    @property
    def fuel_volume_fraction(self) -> float:
        """
        Calculate fuel volume fraction in the assembly.

        Returns:
            Volume fraction of fuel
        """
        fuel_radius = self.cladding.inner_radius - 0.008  # Gap
        fuel_area = math.pi * fuel_radius**2 * self.fuel_pin_count
        total_area = self.assembly_pitch**2
        return fuel_area / total_area

    @property
    def moderator_volume_fraction(self) -> float:
        """
        Calculate moderator volume fraction in the assembly.

        Returns:
            Volume fraction of moderator
        """
        return 1.0 - self.fuel_volume_fraction - self.cladding_volume_fraction

    @property
    def cladding_volume_fraction(self) -> float:
        """
        Calculate cladding volume fraction in the assembly.

        Returns:
            Volume fraction of cladding
        """
        inner_r = self.cladding.inner_radius
        outer_r = self.cladding.outer_radius
        clad_area = math.pi * (outer_r**2 - inner_r**2) * self.fuel_pin_count
        total_area = self.assembly_pitch**2
        return clad_area / total_area

    def get_homogenized_sigma_a(self) -> float:
        """
        Calculate volume-weighted homogenized absorption cross-section.

        Returns:
            Homogenized Σ_a [1/cm]
        """
        return (
            self.fuel_volume_fraction * self.fuel.get_macroscopic_absorption_xs() +
            self.moderator_volume_fraction * self.moderator.get_macroscopic_absorption_xs() +
            self.cladding_volume_fraction * self.cladding.get_macroscopic_absorption_xs()
        )

    def get_homogenized_nu_sigma_f(self) -> float:
        """
        Calculate volume-weighted homogenized νΣ_f.

        Returns:
            Homogenized νΣ_f [1/cm]
        """
        return self.fuel_volume_fraction * self.fuel.get_nu_sigma_f()

    def get_moderation_ratio(self) -> float:
        """
        Calculate hydrogen-to-uranium atom ratio (H/U).

        This is a key parameter for PWR lattice physics.
        Optimal H/U for PWRs is typically around 3-4.

        Returns:
            H/U ratio
        """
        n_h = self.moderator.n_hydrogen * self.moderator_volume_fraction
        n_u = self.fuel.uranium_density * self.fuel_volume_fraction
        return n_h / n_u
