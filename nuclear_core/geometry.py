"""
Core Geometry Module for PWR Modeling

This module defines the physical geometry of the reactor core,
including fuel pins, assemblies, and the overall core configuration.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math
import numpy as np


@dataclass
class FuelPin:
    """
    Single fuel pin geometry specification.

    A fuel pin consists of:
    - UO2 fuel pellets (stacked)
    - Helium-filled gap
    - Zircaloy cladding

    Attributes:
        fuel_radius: Outer radius of fuel pellet [cm]
        gap_thickness: Fuel-clad gap thickness [cm]
        clad_inner_radius: Inner radius of cladding [cm]
        clad_outer_radius: Outer radius of cladding [cm]
        active_length: Active fuel length [cm]
    """

    fuel_radius: float = 0.4096  # [cm]
    gap_thickness: float = 0.0082  # [cm]
    clad_inner_radius: float = 0.4178  # [cm]
    clad_outer_radius: float = 0.475  # [cm]
    active_length: float = 365.76  # [cm] (12 feet)

    def __post_init__(self):
        """Validate geometry consistency."""
        expected_inner = self.fuel_radius + self.gap_thickness
        if abs(self.clad_inner_radius - expected_inner) > 0.001:
            # Allow small tolerance for manufacturing
            pass

        if self.clad_outer_radius <= self.clad_inner_radius:
            raise ValueError("Cladding outer radius must be > inner radius")

    @property
    def fuel_area(self) -> float:
        """Cross-sectional area of fuel [cm²]."""
        return math.pi * self.fuel_radius**2

    @property
    def gap_area(self) -> float:
        """Cross-sectional area of gap [cm²]."""
        return math.pi * (self.clad_inner_radius**2 - self.fuel_radius**2)

    @property
    def clad_area(self) -> float:
        """Cross-sectional area of cladding [cm²]."""
        return math.pi * (self.clad_outer_radius**2 - self.clad_inner_radius**2)

    @property
    def fuel_volume(self) -> float:
        """Volume of fuel in the pin [cm³]."""
        return self.fuel_area * self.active_length

    @property
    def clad_thickness(self) -> float:
        """Cladding wall thickness [cm]."""
        return self.clad_outer_radius - self.clad_inner_radius


@dataclass
class CoreGeometry:
    """
    Full reactor core geometry specification.

    A typical 3000 MWth PWR core consists of ~193 fuel assemblies
    arranged in a roughly cylindrical pattern.

    Attributes:
        thermal_power: Thermal power [MW]
        num_assemblies: Number of fuel assemblies
        assembly_pitch: Distance between assembly centers [cm]
        core_height: Active core height [cm]
        core_barrel_inner_radius: Inner radius of core barrel [cm]
        reflector_thickness: Radial reflector thickness [cm]
    """

    thermal_power: float = 3000.0  # [MW]
    num_assemblies: int = 193
    assembly_pitch: float = 21.504  # [cm] (17 × 1.26 cm)
    core_height: float = 365.76  # [cm]
    core_barrel_inner_radius: float = 187.96  # [cm]
    reflector_thickness: float = 15.0  # [cm] (water reflector)
    fuel_pin: FuelPin = field(default_factory=FuelPin)
    pins_per_assembly: int = 264
    assemblies_layout: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize assembly layout if not provided."""
        if self.assemblies_layout is None:
            self.assemblies_layout = self._generate_assembly_layout()

    def _generate_assembly_layout(self) -> np.ndarray:
        """
        Generate the core assembly layout map.

        Creates a 2D array representing assembly positions.
        Values: 1 = assembly present, 0 = no assembly

        Returns:
            2D numpy array of assembly positions
        """
        # Determine grid size (approximate)
        grid_size = int(math.ceil(math.sqrt(self.num_assemblies * 1.3)))
        if grid_size % 2 == 0:
            grid_size += 1  # Ensure odd for symmetry

        layout = np.zeros((grid_size, grid_size), dtype=int)
        center = grid_size // 2

        # Calculate effective core radius in assembly pitches
        core_radius_assemblies = self.equivalent_radius / self.assembly_pitch

        # Fill assemblies within effective radius
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                # Distance from center in assembly units
                di = i - center
                dj = j - center
                dist = math.sqrt(di**2 + dj**2)

                if dist <= core_radius_assemblies and count < self.num_assemblies:
                    layout[i, j] = 1
                    count += 1

        return layout

    @property
    def equivalent_radius(self) -> float:
        """
        Calculate equivalent cylindrical core radius.

        R_eq = sqrt(N * P² / π)

        where N is number of assemblies and P is assembly pitch.

        Returns:
            Equivalent radius [cm]
        """
        return math.sqrt(self.num_assemblies * self.assembly_pitch**2 / math.pi)

    @property
    def core_volume(self) -> float:
        """
        Calculate active core volume.

        Returns:
            Core volume [cm³]
        """
        return math.pi * self.equivalent_radius**2 * self.core_height

    @property
    def total_fuel_pins(self) -> int:
        """Total number of fuel pins in the core."""
        return self.num_assemblies * self.pins_per_assembly

    @property
    def total_fuel_volume(self) -> float:
        """
        Calculate total fuel volume in the core.

        Returns:
            Total fuel volume [cm³]
        """
        return self.total_fuel_pins * self.fuel_pin.fuel_volume

    @property
    def total_fuel_mass(self) -> float:
        """
        Calculate total fuel mass (assuming 95% theoretical density).

        Returns:
            Total UO2 mass [kg]
        """
        density = 10.97 * 0.95  # [g/cm³]
        return self.total_fuel_volume * density / 1000.0  # [kg]

    @property
    def power_density(self) -> float:
        """
        Calculate volumetric power density.

        Returns:
            Power density [MW/m³] or [W/cm³]
        """
        return self.thermal_power / (self.core_volume * 1e-6)  # MW/m³

    @property
    def linear_power_density(self) -> float:
        """
        Calculate average linear power density.

        Returns:
            Linear power density [kW/m]
        """
        total_fuel_length = self.total_fuel_pins * self.fuel_pin.active_length  # [cm]
        return self.thermal_power * 1000 / (total_fuel_length / 100)  # kW/m

    @property
    def specific_power(self) -> float:
        """
        Calculate specific power (power per unit fuel mass).

        Returns:
            Specific power [MW/tHM] (megawatts per tonne heavy metal)
        """
        # Heavy metal mass (uranium only, not oxygen)
        hm_mass = self.total_fuel_mass * (238.0 / 270.0)  # Approximate U/UO2 ratio
        return self.thermal_power / (hm_mass / 1000.0)

    def get_buckling_geometric(self) -> Tuple[float, float, float]:
        """
        Calculate geometric buckling for a finite cylinder.

        B²_g = B²_r + B²_z = (2.405/R_ex)² + (π/H_ex)²

        where R_ex and H_ex include extrapolation lengths.

        Returns:
            Tuple of (B²_radial, B²_axial, B²_total) [1/cm²]
        """
        # Extrapolation length (approximately 2.13 * D)
        # Using typical value for water-reflected core
        extrap_radial = 7.0  # [cm]
        extrap_axial = 7.0  # [cm]

        r_extrapolated = self.equivalent_radius + extrap_radial
        h_extrapolated = self.core_height + 2 * extrap_axial

        b_sq_radial = (2.405 / r_extrapolated)**2
        b_sq_axial = (math.pi / h_extrapolated)**2
        b_sq_total = b_sq_radial + b_sq_axial

        return b_sq_radial, b_sq_axial, b_sq_total

    def get_flux_shape_radial(self, r: np.ndarray) -> np.ndarray:
        """
        Calculate normalized radial flux shape (J0 Bessel function).

        φ(r) = J0(2.405 * r / R_ex)

        Args:
            r: Radial positions [cm]

        Returns:
            Normalized flux values at each position
        """
        from scipy.special import j0

        r_ex = self.equivalent_radius + 7.0  # Include extrapolation
        return j0(2.405 * r / r_ex)

    def get_flux_shape_axial(self, z: np.ndarray) -> np.ndarray:
        """
        Calculate normalized axial flux shape (cosine).

        φ(z) = cos(π * z / H_ex)

        where z is measured from core center.

        Args:
            z: Axial positions from center [cm]

        Returns:
            Normalized flux values at each position
        """
        h_ex = self.core_height + 14.0  # Include extrapolation
        return np.cos(math.pi * z / h_ex)

    def get_peaking_factors(self) -> dict:
        """
        Calculate flux peaking factors.

        Returns:
            Dictionary with radial, axial, and total peaking factors
        """
        # Analytical values for fundamental mode
        f_radial = 2.32   # For J0 distribution
        f_axial = 1.57    # For cosine distribution (π/2)

        return {
            "radial": f_radial,
            "axial": f_axial,
            "total": f_radial * f_axial,
            "hot_channel_factor": f_radial * f_axial * 1.05  # Engineering factor
        }


@dataclass
class ReflectorGeometry:
    """
    Reflector region geometry.

    The reflector surrounds the active core and reflects
    neutrons back, improving neutron economy.

    Attributes:
        inner_radius: Inner radius (at core boundary) [cm]
        thickness: Radial thickness [cm]
        material: Reflector material type
    """

    inner_radius: float = 168.0  # [cm]
    thickness: float = 20.0  # [cm]
    material: str = "water"

    @property
    def outer_radius(self) -> float:
        """Outer radius of reflector [cm]."""
        return self.inner_radius + self.thickness

    @property
    def volume(self) -> float:
        """Reflector volume (cylindrical shell) [cm³]."""
        # Simplified: cylindrical shell, ignoring top/bottom
        height = 365.76  # [cm]
        return math.pi * (self.outer_radius**2 - self.inner_radius**2) * height

    def get_reflector_savings(self) -> float:
        """
        Estimate reflector savings (δ).

        Reflector savings represents the effective increase in
        core size due to the reflector.

        Returns:
            Reflector savings [cm]
        """
        if self.material == "water":
            # Approximate formula for water reflector
            # δ ≈ 7.2 + 0.1 * thickness (cm) for water
            return min(7.2 + 0.1 * self.thickness, 10.0)
        else:
            return 5.0  # Generic value


def create_standard_pwr_geometry(power_mw: float = 3000.0) -> CoreGeometry:
    """
    Create standard PWR core geometry scaled to specified power.

    Args:
        power_mw: Thermal power in MW

    Returns:
        CoreGeometry instance for specified power level
    """
    # Scale assemblies based on power (roughly linear)
    # Reference: 3000 MW -> 193 assemblies
    num_assemblies = int(193 * power_mw / 3000.0)

    # Ensure reasonable bounds
    num_assemblies = max(100, min(num_assemblies, 250))

    return CoreGeometry(
        thermal_power=power_mw,
        num_assemblies=num_assemblies,
    )
