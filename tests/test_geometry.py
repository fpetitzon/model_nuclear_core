"""
Tests for the geometry module.
"""

import unittest
import math
import numpy as np

from nuclear_core.geometry import (
    FuelPin,
    CoreGeometry,
    ReflectorGeometry,
    create_standard_pwr_geometry,
)


class TestFuelPin(unittest.TestCase):
    """Test fuel pin geometry."""

    def setUp(self):
        self.pin = FuelPin()

    def test_fuel_radius(self):
        """Test fuel pellet radius is reasonable."""
        # Typical UO2 pellet radius is ~0.4 cm
        self.assertTrue(0.3 < self.pin.fuel_radius < 0.5)

    def test_cladding_radius(self):
        """Test cladding outer radius is larger than inner."""
        self.assertTrue(self.pin.clad_outer_radius > self.pin.clad_inner_radius)

    def test_cladding_thickness(self):
        """Test cladding thickness is reasonable."""
        # Typical thickness is ~0.6 mm
        self.assertTrue(0.04 < self.pin.clad_thickness < 0.08)

    def test_active_length(self):
        """Test active fuel length."""
        # Typical 12 ft active length = 365.76 cm
        self.assertAlmostEqual(self.pin.active_length, 365.76, places=0)

    def test_fuel_area(self):
        """Test fuel cross-sectional area."""
        expected = math.pi * self.pin.fuel_radius**2
        self.assertAlmostEqual(self.pin.fuel_area, expected)

    def test_clad_area(self):
        """Test cladding cross-sectional area."""
        r_out = self.pin.clad_outer_radius
        r_in = self.pin.clad_inner_radius
        expected = math.pi * (r_out**2 - r_in**2)
        self.assertAlmostEqual(self.pin.clad_area, expected)

    def test_fuel_volume(self):
        """Test fuel volume calculation."""
        expected = self.pin.fuel_area * self.pin.active_length
        self.assertAlmostEqual(self.pin.fuel_volume, expected)


class TestCoreGeometry(unittest.TestCase):
    """Test core geometry."""

    def setUp(self):
        self.geometry = CoreGeometry()

    def test_default_power(self):
        """Test default thermal power is 3000 MW."""
        self.assertEqual(self.geometry.thermal_power, 3000.0)

    def test_default_assemblies(self):
        """Test default number of assemblies is 193."""
        self.assertEqual(self.geometry.num_assemblies, 193)

    def test_core_height(self):
        """Test core height is 12 feet."""
        self.assertAlmostEqual(self.geometry.core_height, 365.76, places=0)

    def test_total_fuel_pins(self):
        """Test total fuel pin count."""
        expected = self.geometry.num_assemblies * self.geometry.pins_per_assembly
        self.assertEqual(self.geometry.total_fuel_pins, expected)

    def test_equivalent_radius(self):
        """Test equivalent radius calculation."""
        # R = sqrt(N * P² / π)
        N = self.geometry.num_assemblies
        P = self.geometry.assembly_pitch
        expected = math.sqrt(N * P**2 / math.pi)
        self.assertAlmostEqual(self.geometry.equivalent_radius, expected)

    def test_core_volume(self):
        """Test core volume calculation."""
        R = self.geometry.equivalent_radius
        H = self.geometry.core_height
        expected = math.pi * R**2 * H
        self.assertAlmostEqual(self.geometry.core_volume, expected)

    def test_total_fuel_volume(self):
        """Test total fuel volume is positive."""
        self.assertTrue(self.geometry.total_fuel_volume > 0)

    def test_total_fuel_mass(self):
        """Test total fuel mass is reasonable."""
        # Typical 3000 MW PWR has ~100 tonnes of UO2
        mass_tonnes = self.geometry.total_fuel_mass / 1000
        self.assertTrue(80 < mass_tonnes < 150)

    def test_power_density(self):
        """Test power density is positive."""
        self.assertTrue(self.geometry.power_density > 0)

    def test_linear_power_density(self):
        """Test linear power density is reasonable."""
        # Typical PWR average linear power is 15-20 kW/m
        self.assertTrue(10 < self.geometry.linear_power_density < 30)

    def test_specific_power(self):
        """Test specific power is reasonable."""
        # Typical PWR specific power is ~30-40 MW/tHM
        self.assertTrue(20 < self.geometry.specific_power < 50)

    def test_buckling_geometric(self):
        """Test geometric buckling calculation."""
        b_r, b_z, b_total = self.geometry.get_buckling_geometric()

        # All should be positive
        self.assertTrue(b_r > 0)
        self.assertTrue(b_z > 0)
        self.assertTrue(b_total > 0)

        # Total should be sum
        self.assertAlmostEqual(b_total, b_r + b_z, places=10)

    def test_peaking_factors(self):
        """Test peaking factors are reasonable."""
        peaking = self.geometry.get_peaking_factors()

        # Radial peaking for J0 distribution should be ~2.3
        self.assertTrue(2.0 < peaking["radial"] < 2.5)

        # Axial peaking for cosine should be π/2 ≈ 1.57
        self.assertTrue(1.4 < peaking["axial"] < 1.8)

        # Total should be product
        self.assertAlmostEqual(
            peaking["total"],
            peaking["radial"] * peaking["axial"]
        )

    def test_flux_shape_radial(self):
        """Test radial flux shape (Bessel function)."""
        r = np.array([0, 50, 100, 150])
        flux = self.geometry.get_flux_shape_radial(r)

        # Flux at center should be maximum (normalized to 1)
        self.assertAlmostEqual(flux[0], 1.0, places=5)

        # Flux should decrease with radius
        for i in range(len(flux) - 1):
            self.assertTrue(flux[i] >= flux[i+1])

    def test_flux_shape_axial(self):
        """Test axial flux shape (cosine)."""
        z = np.array([0, 50, 100, 150])  # From center
        flux = self.geometry.get_flux_shape_axial(z)

        # Flux at center should be maximum
        self.assertAlmostEqual(flux[0], 1.0, places=5)

        # Flux should decrease with distance from center
        for i in range(len(flux) - 1):
            self.assertTrue(flux[i] >= flux[i+1])


class TestReflectorGeometry(unittest.TestCase):
    """Test reflector geometry."""

    def setUp(self):
        self.reflector = ReflectorGeometry()

    def test_outer_radius(self):
        """Test outer radius calculation."""
        expected = self.reflector.inner_radius + self.reflector.thickness
        self.assertAlmostEqual(self.reflector.outer_radius, expected)

    def test_volume(self):
        """Test reflector volume is positive."""
        self.assertTrue(self.reflector.volume > 0)

    def test_reflector_savings(self):
        """Test reflector savings is positive."""
        delta = self.reflector.get_reflector_savings()
        self.assertTrue(delta > 0)
        # Typical value 5-10 cm for water
        self.assertTrue(3 < delta < 15)


class TestStandardPWRFactory(unittest.TestCase):
    """Test the factory function."""

    def test_default_3000mw(self):
        """Test default 3000 MW core creation."""
        core = create_standard_pwr_geometry()
        self.assertEqual(core.thermal_power, 3000.0)
        self.assertEqual(core.num_assemblies, 193)

    def test_scaled_power(self):
        """Test power scaling changes assembly count."""
        core_small = create_standard_pwr_geometry(power_mw=2000.0)
        core_large = create_standard_pwr_geometry(power_mw=4000.0)

        self.assertTrue(core_small.num_assemblies < core_large.num_assemblies)


if __name__ == "__main__":
    unittest.main()
