"""
Nuclear Reactor Core Model Package

A comprehensive physics-based model for simulating the core of a
3000 MW Pressurized Water Reactor (PWR).

Modules:
    - constants: Physical constants and nuclear data
    - materials: Fuel and moderator material properties
    - geometry: Core geometry definitions
    - neutronics: Neutron flux and criticality calculations
    - thermal: Thermal-hydraulic calculations
    - reactor: Main reactor model integrating all components
"""

from .constants import PhysicalConstants, NuclearData
from .materials import UO2Fuel, LightWaterModerator, FuelAssembly
from .geometry import CoreGeometry, FuelPin
from .neutronics import NeutronicsModel, CriticalityCalculator
from .thermal import ThermalHydraulics
from .reactor import PWRCore

__version__ = "1.0.0"
__author__ = "Nuclear Engineering Model"

__all__ = [
    "PhysicalConstants",
    "NuclearData",
    "UO2Fuel",
    "LightWaterModerator",
    "FuelAssembly",
    "CoreGeometry",
    "FuelPin",
    "NeutronicsModel",
    "CriticalityCalculator",
    "ThermalHydraulics",
    "PWRCore",
]
