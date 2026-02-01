"""
Utility Functions for Nuclear Reactor Modeling

This module provides helper functions for unit conversions,
data validation, and common calculations.
"""

import math
from typing import Tuple, Union
import numpy as np


def pcm_to_delta_k(pcm: float) -> float:
    """
    Convert reactivity from pcm to Δk/k.

    Args:
        pcm: Reactivity in pcm (percent mille)

    Returns:
        Reactivity as Δk/k
    """
    return pcm * 1e-5


def delta_k_to_pcm(delta_k: float) -> float:
    """
    Convert reactivity from Δk/k to pcm.

    Args:
        delta_k: Reactivity as Δk/k

    Returns:
        Reactivity in pcm
    """
    return delta_k * 1e5


def dollars_to_delta_k(dollars: float, beta: float = 0.0065) -> float:
    """
    Convert reactivity from dollars to Δk/k.

    Args:
        dollars: Reactivity in dollars
        beta: Delayed neutron fraction (default: 0.0065 for U-235)

    Returns:
        Reactivity as Δk/k
    """
    return dollars * beta


def delta_k_to_dollars(delta_k: float, beta: float = 0.0065) -> float:
    """
    Convert reactivity from Δk/k to dollars.

    Args:
        delta_k: Reactivity as Δk/k
        beta: Delayed neutron fraction (default: 0.0065 for U-235)

    Returns:
        Reactivity in dollars
    """
    return delta_k / beta


def celsius_to_kelvin(celsius: float) -> float:
    """Convert temperature from Celsius to Kelvin."""
    return celsius + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert temperature from Kelvin to Celsius."""
    return kelvin - 273.15


def fahrenheit_to_kelvin(fahrenheit: float) -> float:
    """Convert temperature from Fahrenheit to Kelvin."""
    return (fahrenheit - 32) * 5/9 + 273.15


def mev_to_joules(mev: float) -> float:
    """Convert energy from MeV to Joules."""
    return mev * 1.60218e-13


def joules_to_mev(joules: float) -> float:
    """Convert energy from Joules to MeV."""
    return joules / 1.60218e-13


def barns_to_cm2(barns: float) -> float:
    """Convert cross-section from barns to cm²."""
    return barns * 1e-24


def cm2_to_barns(cm2: float) -> float:
    """Convert cross-section from cm² to barns."""
    return cm2 * 1e24


def calculate_period(
    reactivity: float,
    beta: float = 0.0065,
    prompt_lifetime: float = 2e-5,
    lambda_eff: float = 0.08
) -> float:
    """
    Calculate reactor period using point kinetics.

    For small reactivities (ρ << β), the period is dominated by
    delayed neutrons:
    T ≈ β / (λ_eff * ρ)

    For large reactivities (ρ > β), the period becomes very short
    (prompt critical).

    Args:
        reactivity: Reactivity ρ (Δk/k)
        beta: Delayed neutron fraction
        prompt_lifetime: Prompt neutron lifetime [s]
        lambda_eff: Effective delayed neutron decay constant [1/s]

    Returns:
        Reactor period [s]
    """
    if abs(reactivity) < 1e-10:
        return float('inf')

    if reactivity >= beta:
        # Prompt supercritical (very dangerous!)
        return prompt_lifetime / (reactivity - beta)
    elif reactivity > 0:
        # Delayed supercritical (normal operation)
        return (beta - reactivity) / (lambda_eff * reactivity)
    else:
        # Subcritical
        return -(beta - reactivity) / (lambda_eff * abs(reactivity))


def calculate_doubling_time(period: float) -> float:
    """
    Calculate power doubling time from reactor period.

    T_2 = T * ln(2)

    Args:
        period: Reactor period [s]

    Returns:
        Doubling time [s]
    """
    if period <= 0 or math.isinf(period):
        return float('inf')
    return period * math.log(2)


def neutron_velocity(energy_ev: float, mass_amu: float = 1.0087) -> float:
    """
    Calculate neutron velocity from kinetic energy.

    v = sqrt(2 * E / m)

    Args:
        energy_ev: Neutron energy [eV]
        mass_amu: Neutron mass [amu] (default: 1.0087)

    Returns:
        Velocity [m/s]
    """
    # Convert to SI units
    energy_j = energy_ev * 1.60218e-19
    mass_kg = mass_amu * 1.66054e-27

    return math.sqrt(2 * energy_j / mass_kg)


def thermal_velocity(temperature_k: float) -> float:
    """
    Calculate most probable thermal neutron velocity.

    v_p = sqrt(2 * k_B * T / m_n)

    Args:
        temperature_k: Temperature [K]

    Returns:
        Most probable velocity [m/s]
    """
    k_b = 1.380649e-23  # Boltzmann constant [J/K]
    m_n = 1.675e-27     # Neutron mass [kg]

    return math.sqrt(2 * k_b * temperature_k / m_n)


def calculate_lethargy(E_initial: float, E_final: float) -> float:
    """
    Calculate lethargy change.

    u = ln(E_initial / E_final)

    Args:
        E_initial: Initial energy [any units]
        E_final: Final energy [same units]

    Returns:
        Lethargy (dimensionless)
    """
    return math.log(E_initial / E_final)


def average_lethargy_gain(A: float) -> float:
    """
    Calculate average lethargy gain per collision (ξ).

    ξ = 1 + (A-1)²/(2A) * ln((A-1)/(A+1))

    For hydrogen (A=1): ξ = 1

    Args:
        A: Atomic mass number of target nucleus

    Returns:
        Average lethargy gain ξ
    """
    if A < 1.01:
        return 1.0  # Hydrogen

    alpha = ((A - 1) / (A + 1))**2
    return 1 + alpha / (1 - alpha) * math.log(alpha)


def collisions_to_thermalize(A: float, E_fast: float = 2e6, E_thermal: float = 0.025) -> int:
    """
    Estimate number of collisions to thermalize a neutron.

    n = ln(E_fast/E_thermal) / ξ

    Args:
        A: Atomic mass number of moderator
        E_fast: Fast neutron energy [eV]
        E_thermal: Thermal neutron energy [eV]

    Returns:
        Approximate number of collisions
    """
    xi = average_lethargy_gain(A)
    lethargy = calculate_lethargy(E_fast, E_thermal)
    return int(math.ceil(lethargy / xi))


def validate_enrichment(enrichment: float) -> Tuple[bool, str]:
    """
    Validate uranium enrichment for commercial PWR.

    Args:
        enrichment: U-235 enrichment [weight %]

    Returns:
        Tuple of (is_valid, message)
    """
    if enrichment < 0.71:
        return False, "Enrichment below natural uranium (0.71%)"
    elif enrichment < 3.0:
        return False, "Enrichment below typical PWR range (3-5%)"
    elif enrichment <= 5.0:
        return True, "Enrichment in standard PWR range (3-5%)"
    elif enrichment <= 20.0:
        return False, "Low-enriched uranium (LEU) but above PWR range"
    else:
        return False, "Highly enriched uranium (HEU) - not for commercial PWR"


def calculate_burnup(
    power_mw: float,
    time_days: float,
    heavy_metal_tonnes: float
) -> float:
    """
    Calculate fuel burnup.

    Burnup = (Power × Time) / Mass

    Args:
        power_mw: Thermal power [MW]
        time_days: Irradiation time [days]
        heavy_metal_tonnes: Heavy metal mass [tonnes]

    Returns:
        Burnup [MWd/tHM]
    """
    return power_mw * time_days / heavy_metal_tonnes


def format_scientific(value: float, precision: int = 3) -> str:
    """
    Format a number in scientific notation.

    Args:
        value: Number to format
        precision: Number of decimal places

    Returns:
        Formatted string
    """
    if value == 0:
        return "0"
    return f"{value:.{precision}e}"


def interpolate_linear(
    x: float,
    x_data: np.ndarray,
    y_data: np.ndarray
) -> float:
    """
    Perform linear interpolation.

    Args:
        x: Point to interpolate at
        x_data: Known x values
        y_data: Known y values

    Returns:
        Interpolated y value
    """
    return np.interp(x, x_data, y_data)
