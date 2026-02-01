"""
Tests for the reactor module.
"""

import unittest
import json
import tempfile
import os

from nuclear_core.reactor import PWRCore, create_pwr_core


class TestPWRCore(unittest.TestCase):
    """Test PWR core model."""

    def setUp(self):
        self.reactor = PWRCore(
            thermal_power=3000.0,
            enrichment=4.0,
        )

    def test_initialization(self):
        """Test reactor initialization."""
        self.assertEqual(self.reactor.thermal_power, 3000.0)
        self.assertEqual(self.reactor.enrichment, 4.0)

    def test_invalid_enrichment_low(self):
        """Test that low enrichment raises error."""
        with self.assertRaises(ValueError):
            PWRCore(thermal_power=3000.0, enrichment=2.0)

    def test_invalid_enrichment_high(self):
        """Test that high enrichment raises error."""
        with self.assertRaises(ValueError):
            PWRCore(thermal_power=3000.0, enrichment=6.0)

    def test_components_initialized(self):
        """Test that all components are initialized."""
        self.assertIsNotNone(self.reactor.fuel)
        self.assertIsNotNone(self.reactor.moderator)
        self.assertIsNotNone(self.reactor.assembly)
        self.assertIsNotNone(self.reactor.geometry)
        self.assertIsNotNone(self.reactor.neutronics)
        self.assertIsNotNone(self.reactor.thermal)
        self.assertIsNotNone(self.reactor.two_group)

    def test_calculate_criticality(self):
        """Test criticality calculation returns expected structure."""
        crit = self.reactor.calculate_criticality()

        # Check structure
        self.assertIn("four_factor", crit)
        self.assertIn("six_factor", crit)
        self.assertIn("reactivity", crit)
        self.assertIn("buckling", crit)

        # Check four-factor contents
        four = crit["four_factor"]
        self.assertIn("eta", four)
        self.assertIn("epsilon", four)
        self.assertIn("p", four)
        self.assertIn("f", four)
        self.assertIn("k_infinity", four)

        # Check six-factor contents
        six = crit["six_factor"]
        self.assertIn("P_fast", six)
        self.assertIn("P_thermal", six)
        self.assertIn("k_effective", six)

    def test_calculate_neutron_flux(self):
        """Test neutron flux calculation."""
        flux = self.reactor.calculate_neutron_flux()

        self.assertIn("average_flux_n_cm2_s", flux)
        self.assertIn("peak_flux_n_cm2_s", flux)
        self.assertIn("migration_length_cm", flux)

        self.assertTrue(flux["average_flux_n_cm2_s"] > 0)
        self.assertTrue(flux["peak_flux_n_cm2_s"] > flux["average_flux_n_cm2_s"])

    def test_calculate_fission_product_poisoning(self):
        """Test fission product poisoning calculation."""
        poison = self.reactor.calculate_fission_product_poisoning()

        self.assertIn("xenon_135", poison)
        self.assertIn("samarium_149", poison)
        self.assertIn("total_poison_worth_pcm", poison)

        # Both should be negative (poisons)
        self.assertTrue(poison["xenon_135"]["reactivity_delta_k"] < 0)
        self.assertTrue(poison["samarium_149"]["reactivity_delta_k"] < 0)

    def test_calculate_thermal_hydraulics(self):
        """Test thermal-hydraulic calculation."""
        th = self.reactor.calculate_thermal_hydraulics()

        self.assertIn("bulk_conditions", th)
        self.assertIn("heat_transfer", th)
        self.assertIn("temperatures", th)
        self.assertIn("safety_margins", th)
        self.assertIn("pressure_drop", th)

    def test_calculate_core_inventory(self):
        """Test core inventory calculation."""
        inv = self.reactor.calculate_core_inventory()

        self.assertIn("total_fuel_pins", inv)
        self.assertIn("uo2_mass_kg", inv)
        self.assertIn("u235_mass_kg", inv)
        self.assertIn("specific_power_MW_tHM", inv)

        # U-235 mass should be less than total uranium
        self.assertTrue(inv["u235_mass_kg"] < inv["uranium_mass_kg"])

    def test_get_geometry_summary(self):
        """Test geometry summary."""
        geom = self.reactor.get_geometry_summary()

        self.assertIn("core", geom)
        self.assertIn("fuel_pin", geom)
        self.assertIn("assembly", geom)

        self.assertEqual(geom["core"]["thermal_power_MW"], 3000.0)
        self.assertEqual(geom["core"]["num_assemblies"], 193)

    def test_run_full_analysis(self):
        """Test full analysis runs without error."""
        analysis = self.reactor.run_full_analysis()

        self.assertIn("metadata", analysis)
        self.assertIn("geometry", analysis)
        self.assertIn("criticality", analysis)
        self.assertIn("neutron_flux", analysis)
        self.assertIn("fission_product_poisoning", analysis)
        self.assertIn("thermal_hydraulics", analysis)
        self.assertIn("core_inventory", analysis)

    def test_to_json(self):
        """Test JSON export."""
        json_str = self.reactor.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        self.assertIn("metadata", data)

    def test_to_json_file(self):
        """Test JSON export to file."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            filepath = f.name

        try:
            self.reactor.to_json(filepath)

            # File should exist and contain valid JSON
            self.assertTrue(os.path.exists(filepath))

            with open(filepath, 'r') as f:
                data = json.load(f)

            self.assertIn("metadata", data)
        finally:
            os.unlink(filepath)


class TestCreatePWRCore(unittest.TestCase):
    """Test factory function."""

    def test_default_creation(self):
        """Test default reactor creation."""
        reactor = create_pwr_core()
        self.assertEqual(reactor.thermal_power, 3000.0)
        self.assertEqual(reactor.enrichment, 4.0)

    def test_custom_parameters(self):
        """Test custom parameter creation."""
        reactor = create_pwr_core(
            power_mw=2800.0,
            enrichment=4.5,
        )
        self.assertEqual(reactor.thermal_power, 2800.0)
        self.assertEqual(reactor.enrichment, 4.5)


class TestPWRCoreSafetyMargins(unittest.TestCase):
    """Test safety margin calculations."""

    def setUp(self):
        self.reactor = PWRCore(thermal_power=3000.0, enrichment=4.0)

    def test_dnbr_above_limit(self):
        """Test DNBR is above safety limit."""
        th = self.reactor.calculate_thermal_hydraulics()
        dnbr = th["safety_margins"]["minimum_dnbr"]

        # DNBR should be above 1.3 (typical safety limit)
        self.assertTrue(dnbr > 1.0)

    def test_fuel_melt_margin(self):
        """Test fuel melt margin is positive."""
        th = self.reactor.calculate_thermal_hydraulics()
        margin = th["temperatures"]["fuel_melt_margin_K"]

        # Should have positive margin to melting
        self.assertTrue(margin > 0)

    def test_temperatures_reasonable(self):
        """Test temperatures are in expected ranges."""
        th = self.reactor.calculate_thermal_hydraulics()
        temps = th["temperatures"]

        # Fuel centerline should be well below melting (3120 K)
        self.assertTrue(temps["peak_fuel_centerline_K"] < 2500)

        # Clad temperature should be below damage limits
        self.assertTrue(temps["peak_clad_temp_K"] < 800)


class TestPWRCoreConsistency(unittest.TestCase):
    """Test internal consistency of calculations."""

    def setUp(self):
        self.reactor = PWRCore(thermal_power=3000.0, enrichment=4.0)

    def test_k_consistency(self):
        """Test k_infinity > k_effective."""
        crit = self.reactor.calculate_criticality()

        k_inf = crit["four_factor"]["k_infinity"]
        k_eff = crit["six_factor"]["k_effective"]

        # k_eff < k_inf due to leakage
        self.assertTrue(k_eff < k_inf)

    def test_reactivity_consistency(self):
        """Test reactivity calculation is consistent."""
        crit = self.reactor.calculate_criticality()

        k_eff = crit["six_factor"]["k_effective"]
        rho = crit["reactivity"]["delta_k_over_k"]

        # ρ = (k-1)/k
        expected_rho = (k_eff - 1.0) / k_eff
        self.assertAlmostEqual(rho, expected_rho, places=10)

    def test_power_balance(self):
        """Test power-related calculations are consistent."""
        inv = self.reactor.calculate_core_inventory()

        # Verify specific power makes sense
        # specific_power = P / M
        expected_sp = (
            self.reactor.thermal_power /
            (inv["uranium_mass_kg"] / 1000.0)
        )
        self.assertAlmostEqual(
            inv["specific_power_MW_tHM"],
            expected_sp,
            places=0
        )


if __name__ == "__main__":
    unittest.main()
