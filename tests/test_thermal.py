"""
Tests for the thermal module.
"""

import unittest
import numpy as np

from nuclear_core.thermal import (
    CoolantProperties,
    FuelProperties,
    CladProperties,
    ThermalHydraulics,
)
from nuclear_core.geometry import CoreGeometry


class TestCoolantProperties(unittest.TestCase):
    """Test coolant properties."""

    def setUp(self):
        self.coolant = CoolantProperties(temperature=573.0, pressure=15.5)

    def test_density_positive(self):
        """Test density is positive."""
        self.assertTrue(self.coolant.density > 0)

    def test_density_decreases_with_temperature(self):
        """Test density decreases with temperature."""
        coolant_cold = CoolantProperties(temperature=300.0, pressure=15.5)
        coolant_hot = CoolantProperties(temperature=600.0, pressure=15.5)

        self.assertTrue(coolant_cold.density > coolant_hot.density)

    def test_specific_heat_positive(self):
        """Test specific heat is positive."""
        self.assertTrue(self.coolant.specific_heat > 0)
        # Should be around 4000-5000 J/kg/K
        self.assertTrue(3000 < self.coolant.specific_heat < 7000)

    def test_thermal_conductivity_positive(self):
        """Test thermal conductivity is positive."""
        self.assertTrue(self.coolant.thermal_conductivity > 0)

    def test_viscosity_positive(self):
        """Test viscosity is positive."""
        self.assertTrue(self.coolant.dynamic_viscosity > 0)

    def test_prandtl_number(self):
        """Test Prandtl number is reasonable."""
        Pr = self.coolant.prandtl_number
        # Pr for water is typically 1-10
        self.assertTrue(0.1 < Pr < 20)

    def test_saturation_temperature(self):
        """Test saturation temperature is reasonable."""
        T_sat = self.coolant.get_saturation_temperature()
        # At 15.5 MPa, T_sat should be around 618 K (345°C)
        self.assertTrue(600 < T_sat < 640)

    def test_subcooling(self):
        """Test subcooling calculation."""
        subcooling = self.coolant.get_subcooling()
        # Coolant at 573K should be subcooled at 15.5 MPa
        self.assertTrue(subcooling > 0)


class TestFuelProperties(unittest.TestCase):
    """Test fuel properties."""

    def setUp(self):
        self.fuel = FuelProperties(temperature=900.0)

    def test_thermal_conductivity_positive(self):
        """Test thermal conductivity is positive."""
        k = self.fuel.thermal_conductivity
        self.assertTrue(k > 0)
        # UO2 conductivity is typically 2-5 W/m/K
        self.assertTrue(1 < k < 10)

    def test_thermal_conductivity_decreases_with_temp(self):
        """Test conductivity decreases with temperature."""
        fuel_cold = FuelProperties(temperature=500.0)
        fuel_hot = FuelProperties(temperature=1500.0)

        self.assertTrue(fuel_cold.thermal_conductivity > fuel_hot.thermal_conductivity)

    def test_specific_heat_positive(self):
        """Test specific heat is positive."""
        cp = self.fuel.specific_heat
        self.assertTrue(cp > 0)

    def test_melting_temperature(self):
        """Test melting temperature is correct."""
        T_melt = self.fuel.get_melting_temperature()
        # UO2 melts at ~3120 K
        self.assertAlmostEqual(T_melt, 3120.0)


class TestCladProperties(unittest.TestCase):
    """Test cladding properties."""

    def setUp(self):
        self.clad = CladProperties(temperature=620.0)

    def test_thermal_conductivity_positive(self):
        """Test thermal conductivity is positive."""
        k = self.clad.thermal_conductivity
        self.assertTrue(k > 0)
        # Zircaloy conductivity is around 12-20 W/m/K
        self.assertTrue(10 < k < 25)


class TestThermalHydraulics(unittest.TestCase):
    """Test thermal-hydraulic calculations."""

    def setUp(self):
        self.geometry = CoreGeometry()
        self.thermal = ThermalHydraulics(
            geometry=self.geometry,
            inlet_temperature=565.0,
            outlet_temperature=598.0,
            system_pressure=15.5,
        )

    def test_core_flow_area_positive(self):
        """Test flow area is positive."""
        self.assertTrue(self.thermal.core_flow_area > 0)

    def test_coolant_velocity_reasonable(self):
        """Test coolant velocity is reasonable."""
        v = self.thermal.coolant_velocity
        # PWR coolant velocity typically 4-6 m/s
        self.assertTrue(2 < v < 10)

    def test_hydraulic_diameter_positive(self):
        """Test hydraulic diameter is positive."""
        D_h = self.thermal.hydraulic_diameter
        self.assertTrue(D_h > 0)
        # Typically around 0.01 m
        self.assertTrue(0.005 < D_h < 0.02)

    def test_reynolds_number(self):
        """Test Reynolds number indicates turbulent flow."""
        Re = self.thermal.calculate_reynolds_number()
        # PWR flow should be highly turbulent (Re > 10^5)
        self.assertTrue(Re > 1e5)

    def test_heat_transfer_coefficient(self):
        """Test heat transfer coefficient is reasonable."""
        h = self.thermal.calculate_heat_transfer_coefficient()
        # Typical PWR h is 20000-50000 W/m²/K
        self.assertTrue(10000 < h < 100000)

    def test_coolant_temperature_profile(self):
        """Test coolant temperature profile."""
        z, T = self.thermal.calculate_coolant_temperature_profile(z_points=50)

        # Should have correct length
        self.assertEqual(len(z), 50)
        self.assertEqual(len(T), 50)

        # Temperature should increase along channel
        self.assertTrue(T[-1] > T[0])

        # First point should be inlet temperature
        self.assertAlmostEqual(T[0], self.thermal.inlet_temperature)

    def test_fuel_temperature_profile(self):
        """Test fuel temperature profile calculation."""
        linear_power = 20000.0  # W/m
        coolant_temp = 580.0    # K

        temps = self.thermal.calculate_fuel_temperature_profile(
            linear_power, coolant_temp
        )

        # Check all expected keys
        self.assertIn("coolant", temps)
        self.assertIn("clad_outer", temps)
        self.assertIn("clad_inner", temps)
        self.assertIn("fuel_surface", temps)
        self.assertIn("fuel_centerline", temps)

        # Temperature should increase toward center
        self.assertTrue(temps["clad_outer"] > temps["coolant"])
        self.assertTrue(temps["clad_inner"] > temps["clad_outer"])
        self.assertTrue(temps["fuel_surface"] > temps["clad_inner"])
        self.assertTrue(temps["fuel_centerline"] > temps["fuel_surface"])

    def test_dnbr_positive(self):
        """Test DNBR is positive."""
        dnbr = self.thermal.calculate_dnbr(20000.0, 580.0)
        self.assertTrue(dnbr > 0)

    def test_pressure_drop(self):
        """Test pressure drop calculation."""
        dp = self.thermal.calculate_pressure_drop()

        # Check all components
        self.assertIn("friction", dp)
        self.assertIn("gravity", dp)
        self.assertIn("form_losses", dp)
        self.assertIn("acceleration", dp)
        self.assertIn("total", dp)

        # All should be positive
        self.assertTrue(dp["friction"] > 0)
        self.assertTrue(dp["gravity"] > 0)
        self.assertTrue(dp["form_losses"] > 0)

        # Total should be sum
        expected_total = (
            dp["friction"] +
            dp["gravity"] +
            dp["form_losses"] +
            dp["acceleration"]
        )
        self.assertAlmostEqual(dp["total"], expected_total)

    def test_hot_channel_temperatures(self):
        """Test hot channel temperature calculation."""
        hot = self.thermal.calculate_hot_channel_temperatures()

        self.assertIn("peak_linear_power_kW_m", hot)
        self.assertIn("peak_fuel_centerline_temp", hot)
        self.assertIn("minimum_dnbr", hot)
        self.assertIn("fuel_melt_margin", hot)

        # Fuel melt margin should be positive (safe)
        self.assertTrue(hot["fuel_melt_margin"] > 0)

    def test_thermal_summary(self):
        """Test thermal summary generation."""
        summary = self.thermal.get_thermal_summary()

        self.assertIn("thermal_power_MW", summary)
        self.assertIn("reynolds_number", summary)
        self.assertIn("minimum_dnbr", summary)
        self.assertIn("core_pressure_drop_kPa", summary)


if __name__ == "__main__":
    unittest.main()
