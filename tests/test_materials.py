"""
Tests for the materials module.
"""

import unittest
import math

from nuclear_core.materials import (
    UO2Fuel,
    LightWaterModerator,
    ZircaloyCladding,
    FuelAssembly,
)


class TestUO2Fuel(unittest.TestCase):
    """Test UO2 fuel material properties."""

    def setUp(self):
        self.fuel = UO2Fuel(enrichment=4.0)

    def test_enrichment_validation_low(self):
        """Test that enrichment below 3% raises error."""
        with self.assertRaises(ValueError):
            UO2Fuel(enrichment=2.5)

    def test_enrichment_validation_high(self):
        """Test that enrichment above 5% raises error."""
        with self.assertRaises(ValueError):
            UO2Fuel(enrichment=5.5)

    def test_valid_enrichment(self):
        """Test that valid enrichment is accepted."""
        fuel = UO2Fuel(enrichment=3.5)
        self.assertEqual(fuel.enrichment, 3.5)

    def test_density(self):
        """Test fuel density calculation."""
        # 95% of theoretical density (10.97 g/cm³)
        expected = 10.97 * 0.95
        self.assertAlmostEqual(self.fuel.density, expected)

    def test_molecular_weight(self):
        """Test UO2 molecular weight."""
        # Should be approximately 270 (238 + 2*16)
        self.assertTrue(265 < self.fuel.molecular_weight < 275)

    def test_number_densities(self):
        """Test uranium number densities."""
        # Number densities should be positive
        self.assertTrue(self.fuel.n_u235 > 0)
        self.assertTrue(self.fuel.n_u238 > 0)
        self.assertTrue(self.fuel.n_oxygen > 0)

        # U-238 should be much more abundant than U-235
        self.assertTrue(self.fuel.n_u238 > self.fuel.n_u235 * 10)

        # Oxygen should be twice uranium
        n_u_total = self.fuel.n_u235 + self.fuel.n_u238
        self.assertAlmostEqual(
            self.fuel.n_oxygen / n_u_total,
            2.0,
            places=3
        )

    def test_macroscopic_fission_xs(self):
        """Test macroscopic fission cross-section is positive."""
        sigma_f = self.fuel.get_macroscopic_fission_xs()
        self.assertTrue(sigma_f > 0)

    def test_macroscopic_absorption_xs(self):
        """Test macroscopic absorption cross-section is positive."""
        sigma_a = self.fuel.get_macroscopic_absorption_xs()
        self.assertTrue(sigma_a > 0)

    def test_nu_sigma_f(self):
        """Test νΣf is positive and reasonable."""
        nu_sigma_f = self.fuel.get_nu_sigma_f()
        self.assertTrue(nu_sigma_f > 0)

    def test_enrichment_effect_on_fission(self):
        """Test that higher enrichment gives higher fission XS."""
        fuel_low = UO2Fuel(enrichment=3.0)
        fuel_high = UO2Fuel(enrichment=5.0)

        sigma_f_low = fuel_low.get_macroscopic_fission_xs()
        sigma_f_high = fuel_high.get_macroscopic_fission_xs()

        self.assertTrue(sigma_f_high > sigma_f_low)


class TestLightWaterModerator(unittest.TestCase):
    """Test light water moderator properties."""

    def setUp(self):
        self.moderator = LightWaterModerator(temperature=573.0, pressure=15.5)

    def test_density_at_operating_conditions(self):
        """Test water density at PWR operating conditions."""
        # Should be around 0.7 g/cm³ at 300°C
        self.assertTrue(0.5 < self.moderator.density < 0.9)

    def test_number_densities(self):
        """Test hydrogen and oxygen number densities."""
        # Hydrogen should be twice oxygen
        ratio = self.moderator.n_hydrogen / self.moderator.n_oxygen
        self.assertAlmostEqual(ratio, 2.0, places=3)

    def test_macroscopic_absorption_xs(self):
        """Test macroscopic absorption cross-section."""
        sigma_a = self.moderator.get_macroscopic_absorption_xs()
        self.assertTrue(sigma_a > 0)

    def test_macroscopic_scattering_xs(self):
        """Test macroscopic scattering cross-section."""
        sigma_s = self.moderator.get_macroscopic_scattering_xs()
        self.assertTrue(sigma_s > 0)
        # Scattering should be much larger than absorption for water
        self.assertTrue(sigma_s > self.moderator.get_macroscopic_absorption_xs() * 10)

    def test_slowing_down_power(self):
        """Test slowing-down power is positive."""
        xi_sigma_s = self.moderator.get_slowing_down_power()
        self.assertTrue(xi_sigma_s > 0)

    def test_moderating_ratio(self):
        """Test moderating ratio is high for water."""
        MR = self.moderator.get_moderating_ratio()
        # Water has excellent moderating ratio (typically > 50)
        self.assertTrue(MR > 30)

    def test_diffusion_coefficient(self):
        """Test diffusion coefficient is positive."""
        D = self.moderator.get_diffusion_coefficient()
        self.assertTrue(D > 0)
        # Typical value around 0.1-0.3 cm
        self.assertTrue(0.05 < D < 0.5)

    def test_diffusion_length(self):
        """Test diffusion length is reasonable."""
        L = self.moderator.get_diffusion_length()
        self.assertTrue(L > 0)
        # Typical value around 2-3 cm for water
        self.assertTrue(1.0 < L < 5.0)


class TestZircaloyCladding(unittest.TestCase):
    """Test Zircaloy cladding properties."""

    def setUp(self):
        self.cladding = ZircaloyCladding()

    def test_inner_radius(self):
        """Test inner radius calculation."""
        expected = self.cladding.outer_radius - self.cladding.thickness
        self.assertAlmostEqual(self.cladding.inner_radius, expected)

    def test_number_density(self):
        """Test Zr number density is positive."""
        self.assertTrue(self.cladding.n_zr > 0)

    def test_low_absorption(self):
        """Test cladding has low absorption cross-section."""
        sigma_a = self.cladding.get_macroscopic_absorption_xs()
        self.assertTrue(sigma_a > 0)
        # Zircaloy has low absorption (neutronically favorable)
        self.assertTrue(sigma_a < 0.01)


class TestFuelAssembly(unittest.TestCase):
    """Test fuel assembly properties."""

    def setUp(self):
        self.assembly = FuelAssembly()

    def test_assembly_configuration(self):
        """Test 17x17 assembly configuration."""
        self.assertEqual(self.assembly.array_size, 17)
        # 17x17 = 289, minus 25 guide tubes = 264 fuel pins
        self.assertEqual(self.assembly.fuel_pin_count, 264)

    def test_volume_fractions_sum_to_one(self):
        """Test that volume fractions sum to approximately 1."""
        total = (
            self.assembly.fuel_volume_fraction +
            self.assembly.moderator_volume_fraction +
            self.assembly.cladding_volume_fraction
        )
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_moderator_dominates(self):
        """Test that moderator has largest volume fraction."""
        self.assertTrue(
            self.assembly.moderator_volume_fraction >
            self.assembly.fuel_volume_fraction
        )

    def test_homogenized_sigma_a(self):
        """Test homogenized absorption cross-section."""
        sigma_a = self.assembly.get_homogenized_sigma_a()
        self.assertTrue(sigma_a > 0)

    def test_homogenized_nu_sigma_f(self):
        """Test homogenized νΣf."""
        nu_sigma_f = self.assembly.get_homogenized_nu_sigma_f()
        self.assertTrue(nu_sigma_f > 0)

    def test_moderation_ratio(self):
        """Test H/U ratio is in expected range."""
        hu_ratio = self.assembly.get_moderation_ratio()
        # Typical PWR H/U ratio is 3-5
        self.assertTrue(2 < hu_ratio < 8)


if __name__ == "__main__":
    unittest.main()
