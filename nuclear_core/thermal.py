"""
Thermal-Hydraulics Module for PWR Core Modeling

This module implements thermal-hydraulic calculations for the reactor core:
- Coolant temperature distribution
- Fuel temperature profiles
- Heat transfer coefficients
- Pressure drop calculations
- Departure from Nucleate Boiling Ratio (DNBR)

The calculations follow standard PWR thermal-hydraulic methodology.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import math
import numpy as np

from .constants import PhysicalConstants
from .geometry import CoreGeometry, FuelPin


@dataclass
class CoolantProperties:
    """
    Water coolant thermophysical properties at PWR conditions.

    Properties are calculated using simplified correlations valid
    for subcooled water at PWR operating conditions.
    """

    temperature: float = 573.0  # [K] (~300°C)
    pressure: float = 15.5  # [MPa]

    @property
    def density(self) -> float:
        """
        Calculate water density [kg/m³].

        Uses simplified correlation for subcooled water.
        """
        # Reference: IAPWS-IF97 (simplified)
        T = self.temperature
        P = self.pressure

        # Approximate correlation for PWR conditions
        rho = 1000.0 * (
            1.0 - 4.5e-4 * (T - 293.0) +
            4.5e-4 * (P - 0.1)
        )

        return max(rho, 500.0)  # Physical lower bound

    @property
    def specific_heat(self) -> float:
        """
        Calculate specific heat capacity [J/kg/K].
        """
        # Approximate value for water at PWR conditions
        # Increases with temperature
        T = self.temperature
        cp = 4200.0 + 2.0 * (T - 293.0)
        return min(cp, 6000.0)

    @property
    def thermal_conductivity(self) -> float:
        """
        Calculate thermal conductivity [W/m/K].
        """
        # Simplified correlation
        T = self.temperature
        k = 0.6 - 0.001 * (T - 293.0)
        return max(k, 0.4)

    @property
    def dynamic_viscosity(self) -> float:
        """
        Calculate dynamic viscosity [Pa·s].
        """
        T = self.temperature
        # Arrhenius-type correlation
        mu = 0.001 * math.exp(-0.02 * (T - 293.0))
        return max(mu, 8e-5)

    @property
    def prandtl_number(self) -> float:
        """Calculate Prandtl number."""
        return self.specific_heat * self.dynamic_viscosity / self.thermal_conductivity

    def get_saturation_temperature(self) -> float:
        """
        Calculate saturation temperature at operating pressure [K].

        Uses linear fit to steam table data for 10-20 MPa range.
        Reference: Steam tables (IAPWS-IF97)
        """
        P = self.pressure  # [MPa]
        # Linear fit from steam tables (10-20 MPa range)
        # At 10 MPa: T_sat = 584.1 K
        # At 15 MPa: T_sat = 615.3 K
        # At 20 MPa: T_sat = 638.9 K
        T_sat = 584.1 + 5.48 * (P - 10.0)
        return T_sat

    def get_subcooling(self) -> float:
        """
        Calculate subcooling margin [K].
        """
        return self.get_saturation_temperature() - self.temperature


@dataclass
class FuelProperties:
    """
    UO2 fuel thermophysical properties.
    """

    temperature: float = 900.0  # [K] average fuel temperature
    density: float = 10420.0  # [kg/m³] at 95% TD

    @property
    def thermal_conductivity(self) -> float:
        """
        Calculate UO2 thermal conductivity [W/m/K].

        Uses MATPRO correlation for unirradiated UO2.
        """
        T = self.temperature
        # MATPRO correlation (simplified)
        k = 1.0 / (0.0375 + 2.165e-4 * T) + 4.715e9 * math.exp(-16350.0 / T) / T**2
        return k

    @property
    def specific_heat(self) -> float:
        """
        Calculate UO2 specific heat [J/kg/K].
        """
        T = self.temperature
        # Polynomial fit
        cp = 235.0 + 0.135 * T - 3.5e-5 * T**2
        return cp

    def get_melting_temperature(self) -> float:
        """UO2 melting temperature [K]."""
        return 3120.0  # Fresh UO2


@dataclass
class CladProperties:
    """
    Zircaloy cladding thermophysical properties.
    """

    temperature: float = 620.0  # [K]
    density: float = 6550.0  # [kg/m³]

    @property
    def thermal_conductivity(self) -> float:
        """
        Calculate Zircaloy thermal conductivity [W/m/K].
        """
        T = self.temperature
        # Linear approximation
        return 12.0 + 0.008 * T


@dataclass
class ThermalHydraulics:
    """
    Complete thermal-hydraulic model for PWR core.

    Calculates temperature distributions, heat transfer coefficients,
    and safety margins (DNBR, fuel centerline temperature).
    """

    geometry: CoreGeometry
    inlet_temperature: float = 565.0  # [K] (~292°C)
    outlet_temperature: float = 598.0  # [K] (~325°C)
    system_pressure: float = 15.5  # [MPa]
    mass_flow_rate: float = 17400.0  # [kg/s] total core flow
    fuel_pin: FuelPin = field(default_factory=FuelPin)

    def __post_init__(self):
        """Initialize derived quantities."""
        self.coolant_inlet = CoolantProperties(
            temperature=self.inlet_temperature,
            pressure=self.system_pressure
        )
        self.fuel_props = FuelProperties()
        self.clad_props = CladProperties()

    @property
    def core_flow_area(self) -> float:
        """
        Calculate total coolant flow area [m²].
        """
        # Flow area per fuel pin (square lattice minus pin)
        pitch = 0.0126  # [m]
        pin_od = self.fuel_pin.clad_outer_radius * 2 / 100  # [m]
        area_per_pin = pitch**2 - math.pi * (pin_od/2)**2

        return area_per_pin * self.geometry.total_fuel_pins

    @property
    def coolant_velocity(self) -> float:
        """
        Calculate average coolant velocity [m/s].
        """
        rho = self.coolant_inlet.density
        return self.mass_flow_rate / (rho * self.core_flow_area)

    @property
    def hydraulic_diameter(self) -> float:
        """
        Calculate hydraulic diameter [m].

        D_h = 4 * A / P_wetted
        """
        pitch = 0.0126  # [m]
        pin_od = self.fuel_pin.clad_outer_radius * 2 / 100  # [m]

        flow_area = pitch**2 - math.pi * (pin_od/2)**2
        wetted_perimeter = math.pi * pin_od

        return 4 * flow_area / wetted_perimeter

    def calculate_reynolds_number(self, temperature: float = None) -> float:
        """
        Calculate Reynolds number.
        """
        if temperature is None:
            temperature = (self.inlet_temperature + self.outlet_temperature) / 2

        coolant = CoolantProperties(temperature=temperature, pressure=self.system_pressure)

        Re = (
            coolant.density *
            self.coolant_velocity *
            self.hydraulic_diameter /
            coolant.dynamic_viscosity
        )
        return Re

    def calculate_heat_transfer_coefficient(
        self,
        temperature: float = None
    ) -> float:
        """
        Calculate convective heat transfer coefficient [W/m²/K].

        Uses Dittus-Boelter correlation for turbulent flow:
        Nu = 0.023 * Re^0.8 * Pr^0.4
        """
        if temperature is None:
            temperature = (self.inlet_temperature + self.outlet_temperature) / 2

        coolant = CoolantProperties(temperature=temperature, pressure=self.system_pressure)
        Re = self.calculate_reynolds_number(temperature)
        Pr = coolant.prandtl_number

        # Dittus-Boelter correlation
        Nu = 0.023 * Re**0.8 * Pr**0.4

        h = Nu * coolant.thermal_conductivity / self.hydraulic_diameter
        return h

    def calculate_coolant_temperature_profile(
        self,
        z_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate axial coolant temperature distribution.

        Assumes cosine power distribution.

        Args:
            z_points: Number of axial points

        Returns:
            Tuple of (z_positions [m], temperatures [K])
        """
        H = self.geometry.core_height / 100  # [m]
        z = np.linspace(0, H, z_points)

        # Power per pin
        power_per_pin = (
            self.geometry.thermal_power * 1e6 /
            self.geometry.total_fuel_pins
        )  # [W]

        # Mass flow per pin
        m_dot = self.mass_flow_rate / self.geometry.total_fuel_pins

        # Integrate cosine power distribution
        # q'(z) = q'_max * cos(π * (z - H/2) / H_ex)
        H_ex = H + 0.14  # Extrapolated height [m]
        peaking = self.geometry.get_peaking_factors()

        temperatures = np.zeros_like(z)
        temperatures[0] = self.inlet_temperature

        cp = self.coolant_inlet.specific_heat

        for i in range(1, len(z)):
            # Integrated power up to this point
            z_mid = (z[i] + z[i-1]) / 2 - H/2
            dz = z[i] - z[i-1]

            # Local linear power density
            q_prime = (
                power_per_pin / H *
                peaking["axial"] *
                math.cos(math.pi * z_mid / H_ex)
            )

            dT = q_prime * dz / (m_dot * cp)
            temperatures[i] = temperatures[i-1] + dT

        return z, temperatures

    def calculate_fuel_temperature_profile(
        self,
        linear_power: float,
        coolant_temp: float
    ) -> Dict[str, float]:
        """
        Calculate radial temperature profile in fuel pin.

        Args:
            linear_power: Linear power density [W/m]
            coolant_temp: Bulk coolant temperature [K]

        Returns:
            Dictionary with temperatures at key locations
        """
        # Dimensions
        r_fuel = self.fuel_pin.fuel_radius / 100  # [m]
        r_clad_in = self.fuel_pin.clad_inner_radius / 100  # [m]
        r_clad_out = self.fuel_pin.clad_outer_radius / 100  # [m]
        gap = r_clad_in - r_fuel

        # Heat transfer coefficient
        h_conv = self.calculate_heat_transfer_coefficient(coolant_temp)

        # Gap conductance (approximate)
        h_gap = 5000.0  # [W/m²/K] typical value

        # Material properties
        k_fuel = FuelProperties(temperature=900.0).thermal_conductivity
        k_clad = CladProperties(temperature=620.0).thermal_conductivity

        # Heat flux at cladding surface
        q_double_prime = linear_power / (2 * math.pi * r_clad_out)

        # Temperature drops (radial heat conduction)
        # Coolant to clad surface
        dT_conv = q_double_prime / h_conv

        # Through cladding
        dT_clad = (
            linear_power / (2 * math.pi * k_clad) *
            math.log(r_clad_out / r_clad_in)
        )

        # Across gap
        dT_gap = linear_power / (2 * math.pi * r_clad_in * h_gap)

        # Through fuel
        dT_fuel = linear_power / (4 * math.pi * k_fuel)

        # Calculate temperatures
        T_clad_outer = coolant_temp + dT_conv
        T_clad_inner = T_clad_outer + dT_clad
        T_fuel_surface = T_clad_inner + dT_gap
        T_fuel_centerline = T_fuel_surface + dT_fuel

        return {
            "coolant": coolant_temp,
            "clad_outer": T_clad_outer,
            "clad_inner": T_clad_inner,
            "fuel_surface": T_fuel_surface,
            "fuel_centerline": T_fuel_centerline,
        }

    def calculate_dnbr(
        self,
        linear_power: float,
        coolant_temp: float
    ) -> float:
        """
        Calculate Departure from Nucleate Boiling Ratio.

        DNBR = q"_CHF / q"_actual

        Uses W-3 correlation (simplified) for critical heat flux.

        Args:
            linear_power: Local linear power [W/m]
            coolant_temp: Local coolant temperature [K]

        Returns:
            DNBR (should be > 1.3 for safety)
        """
        r_clad_out = self.fuel_pin.clad_outer_radius / 100  # [m]
        q_actual = linear_power / (2 * math.pi * r_clad_out)  # [W/m²]

        # W-3 correlation for CHF (simplified)
        coolant = CoolantProperties(
            temperature=coolant_temp,
            pressure=self.system_pressure
        )

        P = self.system_pressure  # [MPa]
        G = self.mass_flow_rate / self.core_flow_area  # [kg/m²/s]

        # Quality (negative for subcooled)
        T_sat = coolant.get_saturation_temperature()
        h_f = coolant.specific_heat * (T_sat - 273.15)  # Approximate
        h = coolant.specific_heat * (coolant_temp - 273.15)
        h_fg = 1000000.0  # Approximate latent heat [J/kg]
        x_eq = (h - h_f) / h_fg  # Equilibrium quality (negative for subcooled)

        # Simplified W-3 correlation
        # q"_CHF = A * (B + C*x) * G^n
        A = 2.0e6 * (1.0 - 0.04 * (P - 14.0))  # Pressure effect
        B = 1.0
        C = -0.5
        n = 0.4

        q_chf = A * (B + C * x_eq) * (G / 1000.0)**n

        dnbr = q_chf / q_actual

        return dnbr

    def calculate_pressure_drop(self) -> Dict[str, float]:
        """
        Calculate core pressure drop components.

        Returns:
            Dictionary with pressure drop components [kPa]
        """
        H = self.geometry.core_height / 100  # [m]

        # Average properties
        T_avg = (self.inlet_temperature + self.outlet_temperature) / 2
        coolant = CoolantProperties(temperature=T_avg, pressure=self.system_pressure)

        rho = coolant.density
        v = self.coolant_velocity
        Re = self.calculate_reynolds_number(T_avg)
        D_h = self.hydraulic_diameter

        # Friction factor (Blasius correlation for turbulent flow)
        f = 0.316 / Re**0.25

        # Friction pressure drop
        dp_friction = f * (H / D_h) * (rho * v**2 / 2) / 1000  # [kPa]

        # Gravity pressure drop
        dp_gravity = rho * 9.81 * H / 1000  # [kPa]

        # Form losses (grids, etc.) - approximate
        K_total = 10.0  # Total form loss coefficient
        dp_form = K_total * (rho * v**2 / 2) / 1000  # [kPa]

        # Acceleration pressure drop (due to density change)
        rho_in = CoolantProperties(
            temperature=self.inlet_temperature,
            pressure=self.system_pressure
        ).density
        rho_out = CoolantProperties(
            temperature=self.outlet_temperature,
            pressure=self.system_pressure
        ).density

        G = self.mass_flow_rate / self.core_flow_area
        dp_accel = G**2 * (1/rho_out - 1/rho_in) / 1000  # [kPa]

        return {
            "friction": dp_friction,
            "gravity": dp_gravity,
            "form_losses": dp_form,
            "acceleration": dp_accel,
            "total": dp_friction + dp_gravity + dp_form + dp_accel,
        }

    def calculate_hot_channel_temperatures(self) -> Dict[str, float]:
        """
        Calculate temperatures in the hot channel.

        The hot channel is the channel with the highest power,
        accounting for peaking factors.

        Returns:
            Dictionary with peak temperatures
        """
        peaking = self.geometry.get_peaking_factors()

        # Average linear power
        avg_linear_power = (
            self.geometry.thermal_power * 1e6 /
            (self.geometry.total_fuel_pins * self.geometry.core_height / 100)
        )

        # Peak linear power
        peak_linear_power = avg_linear_power * peaking["hot_channel_factor"]

        # Hot channel coolant temperature (at axial peak)
        # Approximate: use average outlet temperature with small increase
        hot_coolant_temp = self.outlet_temperature + 5.0  # [K]

        # Calculate fuel temperatures at hot spot
        temps = self.calculate_fuel_temperature_profile(
            peak_linear_power,
            hot_coolant_temp
        )

        # DNBR at hot spot
        dnbr = self.calculate_dnbr(peak_linear_power, hot_coolant_temp)

        return {
            "peak_linear_power_kW_m": peak_linear_power / 1000,
            "hot_channel_coolant_temp": hot_coolant_temp,
            "peak_clad_temp": temps["clad_outer"],
            "peak_fuel_centerline_temp": temps["fuel_centerline"],
            "minimum_dnbr": dnbr,
            "fuel_melt_margin": FuelProperties().get_melting_temperature() - temps["fuel_centerline"],
        }

    def get_thermal_summary(self) -> Dict[str, float]:
        """
        Get summary of thermal-hydraulic conditions.

        Returns:
            Dictionary with key thermal-hydraulic parameters
        """
        hot_channel = self.calculate_hot_channel_temperatures()
        pressure_drop = self.calculate_pressure_drop()

        # Average temperatures
        T_avg = (self.inlet_temperature + self.outlet_temperature) / 2
        avg_fuel_temps = self.calculate_fuel_temperature_profile(
            self.geometry.thermal_power * 1e6 / (
                self.geometry.total_fuel_pins *
                self.geometry.core_height / 100
            ),
            T_avg
        )

        return {
            "thermal_power_MW": self.geometry.thermal_power,
            "inlet_temperature_K": self.inlet_temperature,
            "outlet_temperature_K": self.outlet_temperature,
            "coolant_velocity_m_s": self.coolant_velocity,
            "mass_flow_rate_kg_s": self.mass_flow_rate,
            "reynolds_number": self.calculate_reynolds_number(),
            "heat_transfer_coeff_W_m2K": self.calculate_heat_transfer_coefficient(),
            "avg_fuel_centerline_K": avg_fuel_temps["fuel_centerline"],
            "peak_fuel_centerline_K": hot_channel["peak_fuel_centerline_temp"],
            "fuel_melt_margin_K": hot_channel["fuel_melt_margin"],
            "minimum_dnbr": hot_channel["minimum_dnbr"],
            "core_pressure_drop_kPa": pressure_drop["total"],
        }
