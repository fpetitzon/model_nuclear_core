"""
Tests for the constants module.
"""

import unittest
import math

from nuclear_core.constants import (
    PhysicalConstants,
    NuclearData,
    DELAYED_NEUTRON_DATA,
    FISSION_PRODUCT_DATA,
    TWO_GROUP_STRUCTURE,
)


class TestPhysicalConstants(unittest.TestCase):
    """Test physical constants."""

    def setUp(self):
        self.constants = PhysicalConstants()

    def test_avogadro_number(self):
        """Test Avogadro's number is correct."""
        self.assertAlmostEqual(
            self.constants.AVOGADRO,
            6.02214076e23,
            places=5
        )

    def test_energy_per_fission(self):
        """Test energy per fission is approximately 200 MeV."""
        self.assertAlmostEqual(
            self.constants.ENERGY_PER_FISSION_MEV,
            200.0,
            places=1
        )

    def test_thermal_neutron_velocity(self):
        """Test thermal neutron velocity is 2200 m/s."""
        self.assertEqual(
            self.constants.THERMAL_NEUTRON_VELOCITY,
            2200.0
        )

    def test_barn_conversion(self):
        """Test barn to cm² conversion factor."""
        self.assertEqual(
            self.constants.BARN_TO_CM2,
            1e-24
        )


class TestNuclearData(unittest.TestCase):
    """Test nuclear cross-section data."""

    def setUp(self):
        self.nuclear_data = NuclearData()

    def test_u235_fission_cross_section(self):
        """Test U-235 thermal fission cross-section is reasonable."""
        sigma_f = self.nuclear_data.u235["sigma_f"]
        # Should be around 585 barns for thermal neutrons
        self.assertTrue(500 < sigma_f < 700)

    def test_u235_nu(self):
        """Test U-235 neutrons per fission."""
        nu = self.nuclear_data.u235["nu"]
        # Should be around 2.4 for thermal fission
        self.assertTrue(2.3 < nu < 2.6)

    def test_u238_resonance_integral(self):
        """Test U-238 resonance integral is reasonable."""
        I = self.nuclear_data.u238["resonance_integral"]
        # Should be around 275 barns
        self.assertTrue(200 < I < 350)

    def test_hydrogen_scattering(self):
        """Test hydrogen scattering cross-section."""
        sigma_s = self.nuclear_data.h1["sigma_s"]
        # Should be around 20 barns
        self.assertTrue(15 < sigma_s < 25)

    def test_u235_absorption_xs(self):
        """Test U-235 absorption cross-section calculation."""
        sigma_a = self.nuclear_data.get_u235_sigma_a()
        # Should be fission + capture
        expected = (
            self.nuclear_data.u235["sigma_f"] +
            self.nuclear_data.u235["sigma_c"]
        )
        self.assertAlmostEqual(sigma_a, expected)

    def test_h2o_absorption_xs(self):
        """Test H2O absorption cross-section calculation."""
        sigma_a = self.nuclear_data.get_h2o_sigma_a()
        # Should be 2*H + O
        expected = (
            2 * self.nuclear_data.h1["sigma_c"] +
            self.nuclear_data.o16["sigma_c"]
        )
        self.assertAlmostEqual(sigma_a, expected)

    def test_temperature_correction(self):
        """Test 1/v temperature correction for cross-sections."""
        sigma_ref = 100.0  # barns
        T_ref = 293.0      # K
        T_high = 573.0     # K (300°C)

        sigma_corrected = self.nuclear_data.temperature_corrected_sigma(
            sigma_ref, T_ref, T_high
        )

        # Cross-section should decrease with temperature
        self.assertTrue(sigma_corrected < sigma_ref)

        # Check the sqrt relationship
        expected = sigma_ref * math.sqrt(T_ref / T_high)
        self.assertAlmostEqual(sigma_corrected, expected)


class TestDelayedNeutronData(unittest.TestCase):
    """Test delayed neutron data."""

    def test_beta_total(self):
        """Test total delayed neutron fraction."""
        beta = DELAYED_NEUTRON_DATA["beta_total"]
        # Should be around 0.0065 for U-235
        self.assertTrue(0.005 < beta < 0.008)

    def test_beta_sum(self):
        """Test that individual betas sum approximately to total."""
        betas = DELAYED_NEUTRON_DATA["betas"]
        total = sum(betas)
        expected = DELAYED_NEUTRON_DATA["beta_total"]
        # Allow some tolerance
        self.assertAlmostEqual(total, expected, places=4)

    def test_six_groups(self):
        """Test that there are 6 delayed neutron groups."""
        self.assertEqual(DELAYED_NEUTRON_DATA["groups"], 6)
        self.assertEqual(len(DELAYED_NEUTRON_DATA["betas"]), 6)
        self.assertEqual(len(DELAYED_NEUTRON_DATA["lambdas"]), 6)


class TestFissionProductData(unittest.TestCase):
    """Test fission product data."""

    def test_xe135_cross_section(self):
        """Test Xe-135 has very large absorption cross-section."""
        sigma_xe = FISSION_PRODUCT_DATA["Xe-135"]["sigma_a"]
        # Xe-135 has one of the largest cross-sections
        self.assertTrue(sigma_xe > 1e6)

    def test_sm149_is_stable(self):
        """Test Sm-149 has zero decay constant (stable)."""
        lambda_sm = FISSION_PRODUCT_DATA["Sm-149"]["decay_constant"]
        self.assertEqual(lambda_sm, 0.0)


class TestTwoGroupStructure(unittest.TestCase):
    """Test two-group energy structure."""

    def test_chi_sum(self):
        """Test that fission spectrum sums to 1."""
        chi_fast = TWO_GROUP_STRUCTURE["fast"]["chi"]
        chi_thermal = TWO_GROUP_STRUCTURE["thermal"]["chi"]
        self.assertAlmostEqual(chi_fast + chi_thermal, 1.0)

    def test_energy_boundary(self):
        """Test energy boundary between groups."""
        e_lower_fast = TWO_GROUP_STRUCTURE["fast"]["E_lower"]
        e_upper_thermal = TWO_GROUP_STRUCTURE["thermal"]["E_upper"]
        self.assertEqual(e_lower_fast, e_upper_thermal)


if __name__ == "__main__":
    unittest.main()
