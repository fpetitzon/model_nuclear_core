"""
Physical Constants and Nuclear Data for PWR Modeling

This module contains fundamental physical constants and nuclear cross-section
data required for neutronics calculations in a Pressurized Water Reactor.
"""

from dataclasses import dataclass
from typing import Dict
import math


@dataclass(frozen=True)
class PhysicalConstants:
    """Fundamental physical constants used in reactor physics calculations."""

    # Avogadro's number [atoms/mol]
    AVOGADRO: float = 6.02214076e23

    # Boltzmann constant [eV/K]
    BOLTZMANN_EV: float = 8.617333262e-5

    # Neutron mass [kg]
    NEUTRON_MASS: float = 1.674927471e-27

    # Speed of light [m/s]
    SPEED_OF_LIGHT: float = 2.99792458e8

    # Energy per fission [MeV] - U-235 average
    ENERGY_PER_FISSION_MEV: float = 200.0

    # Energy per fission [J]
    ENERGY_PER_FISSION_J: float = 3.204e-11

    # Thermal neutron velocity at 20°C [m/s]
    THERMAL_NEUTRON_VELOCITY: float = 2200.0

    # Reference temperature for cross-sections [K]
    REFERENCE_TEMPERATURE: float = 293.6

    # MeV to Joules conversion
    MEV_TO_JOULES: float = 1.60218e-13

    # Barn to cm² conversion
    BARN_TO_CM2: float = 1e-24


class NuclearData:
    """
    Nuclear cross-section data and properties for reactor materials.

    Cross-sections are provided at thermal energy (0.0253 eV) and represent
    microscopic cross-sections in barns.

    Data sources: ENDF/B-VIII.0, JEFF-3.3
    """

    def __init__(self):
        # Thermal neutron energy [eV]
        self.thermal_energy = 0.0253

        # U-235 nuclear data
        self.u235 = {
            "atomic_mass": 235.0439,  # [amu]
            "sigma_f": 585.1,         # Fission cross-section [barns]
            "sigma_c": 98.8,          # Capture cross-section [barns]
            "sigma_s": 15.0,          # Scattering cross-section [barns]
            "nu": 2.432,              # Neutrons per fission
            "eta": 2.068,             # Neutrons per absorption (thermal)
        }

        # U-238 nuclear data
        self.u238 = {
            "atomic_mass": 238.0508,  # [amu]
            "sigma_f": 0.00002,       # Fission cross-section [barns] (threshold)
            "sigma_c": 2.68,          # Capture cross-section [barns]
            "sigma_s": 8.9,           # Scattering cross-section [barns]
            "resonance_integral": 275.0,  # Resonance integral [barns]
        }

        # Oxygen-16 nuclear data
        self.o16 = {
            "atomic_mass": 15.9949,
            "sigma_c": 0.00019,       # Capture cross-section [barns]
            "sigma_s": 3.76,          # Scattering cross-section [barns]
        }

        # Hydrogen-1 nuclear data (for light water)
        self.h1 = {
            "atomic_mass": 1.00794,
            "sigma_c": 0.332,         # Capture cross-section [barns]
            "sigma_s": 20.47,         # Scattering cross-section [barns]
            "xi": 1.0,                # Average lethargy gain per collision
        }

        # Zirconium (cladding) nuclear data
        self.zr = {
            "atomic_mass": 91.224,
            "sigma_c": 0.185,         # Capture cross-section [barns]
            "sigma_s": 6.2,           # Scattering cross-section [barns]
        }

        # Water molecule data
        self.h2o = {
            "molecular_mass": 18.015,
            "density_293K": 0.998,    # [g/cm³] at 20°C
            "density_573K": 0.712,    # [g/cm³] at 300°C (PWR operating)
        }

    def get_u235_sigma_a(self) -> float:
        """Get U-235 absorption cross-section [barns]."""
        return self.u235["sigma_f"] + self.u235["sigma_c"]

    def get_u238_sigma_a(self) -> float:
        """Get U-238 absorption cross-section [barns]."""
        return self.u238["sigma_f"] + self.u238["sigma_c"]

    def get_h2o_sigma_a(self) -> float:
        """Get H2O absorption cross-section [barns]."""
        # Two hydrogen atoms per water molecule
        return 2 * self.h1["sigma_c"] + self.o16["sigma_c"]

    def get_h2o_sigma_s(self) -> float:
        """Get H2O scattering cross-section [barns]."""
        return 2 * self.h1["sigma_s"] + self.o16["sigma_s"]

    def temperature_corrected_sigma(
        self,
        sigma_ref: float,
        temp_ref: float,
        temp_actual: float
    ) -> float:
        """
        Apply 1/v correction for thermal cross-sections.

        For thermal neutrons, absorption cross-sections follow 1/v law:
        σ(T) = σ(T_ref) * sqrt(T_ref / T)

        Args:
            sigma_ref: Reference cross-section [barns]
            temp_ref: Reference temperature [K]
            temp_actual: Actual temperature [K]

        Returns:
            Temperature-corrected cross-section [barns]
        """
        return sigma_ref * math.sqrt(temp_ref / temp_actual)


# Delayed neutron data for U-235
DELAYED_NEUTRON_DATA = {
    "groups": 6,
    "beta_total": 0.0065,  # Total delayed neutron fraction
    "betas": [0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273],
    "lambdas": [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01],  # Decay constants [1/s]
}


# Fission product yields (selected important isotopes)
FISSION_PRODUCT_DATA = {
    "Xe-135": {
        "yield": 0.061,           # Cumulative fission yield
        "sigma_a": 2.65e6,        # Absorption cross-section [barns]
        "decay_constant": 2.09e-5  # [1/s]
    },
    "Sm-149": {
        "yield": 0.0107,
        "sigma_a": 40140,
        "decay_constant": 0.0,    # Stable
    },
    "I-135": {
        "yield": 0.0639,
        "sigma_a": 7.0,
        "decay_constant": 2.87e-5  # [1/s]
    }
}


# Energy group structure for two-group calculations
TWO_GROUP_STRUCTURE = {
    "fast": {
        "E_upper": 10.0e6,     # [eV]
        "E_lower": 0.625,      # [eV]
        "chi": 0.985,          # Fission spectrum fraction
    },
    "thermal": {
        "E_upper": 0.625,      # [eV]
        "E_lower": 0.0,        # [eV]
        "chi": 0.015,          # Fission spectrum fraction
    }
}
