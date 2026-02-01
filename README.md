# Nuclear Reactor Core Model

A comprehensive physics-based model for simulating the core of a 3000 MW Pressurized Water Reactor (PWR). This model implements fundamental reactor physics calculations including neutronics, criticality analysis, and thermal-hydraulics.

## Overview

This package provides a complete reactor core simulation capability including:

- **Neutronics Calculations**
  - Four-factor formula (η, ε, p, f)
  - Six-factor formula (including leakage probabilities P_F, P_T)
  - Effective multiplication factor k_eff
  - Neutron flux distributions (spatial and energy-dependent)
  - Two-group diffusion theory

- **Criticality Analysis**
  - Infinite and effective multiplication factors
  - Reactivity in multiple units (Δk/k, pcm, dollars)
  - Geometric buckling calculations
  - Leakage probability estimates

- **Thermal-Hydraulics**
  - Coolant temperature profiles
  - Fuel and cladding temperature distributions
  - Heat transfer coefficients (Dittus-Boelter correlation)
  - Pressure drop calculations
  - Safety margins (DNBR, fuel melt margin)

- **Fission Product Poisoning**
  - Xe-135 equilibrium concentration and reactivity worth
  - Sm-149 equilibrium poisoning

## Physics Background

### Four-Factor Formula

The infinite multiplication factor k∞ is calculated using the four-factor formula:

```
k∞ = η × ε × p × f
```

Where:
- **η (eta)**: Reproduction factor - average number of fast neutrons produced per thermal neutron absorbed in fuel
- **ε (epsilon)**: Fast fission factor - enhancement due to fast fissions in U-238
- **p**: Resonance escape probability - probability neutrons avoid capture in U-238 resonances
- **f**: Thermal utilization factor - fraction of thermal neutrons absorbed in fuel vs. total

### Six-Factor Formula

For a finite reactor, the effective multiplication factor includes leakage:

```
k_eff = k∞ × P_F × P_T
```

Where:
- **P_F**: Fast non-leakage probability (1 / (1 + τB²))
- **P_T**: Thermal non-leakage probability (1 / (1 + L²B²))
- **τ**: Fermi age (slowing-down length squared)
- **L**: Thermal diffusion length
- **B²**: Geometric buckling

### Neutron Flux Distribution

The fundamental mode flux distribution in a cylindrical core:

```
φ(r,z) = φ_0 × J_0(2.405r/R_ex) × cos(πz/H_ex)
```

Where J_0 is the Bessel function of the first kind and R_ex, H_ex include extrapolation lengths.

## Installation

### Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-repo/nuclear_core.git
cd nuclear_core

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Quick Install

```bash
pip install numpy scipy
```

## Usage

### Basic Usage

```python
from nuclear_core import PWRCore, create_pwr_core

# Create a 3000 MW PWR with 4% enrichment
reactor = create_pwr_core(
    power_mw=3000.0,
    enrichment=4.0,
)

# Run full analysis
analysis = reactor.run_full_analysis()

# Print summary
reactor.print_summary()
```

### Detailed Criticality Analysis

```python
from nuclear_core import PWRCore

reactor = PWRCore(
    thermal_power=3000.0,
    enrichment=4.5,
    coolant_inlet_temp=565.0,
    coolant_outlet_temp=598.0,
    system_pressure=15.5,
)

# Get criticality parameters
criticality = reactor.calculate_criticality()

# Access four-factor components
eta = criticality['four_factor']['eta']
epsilon = criticality['four_factor']['epsilon']
p = criticality['four_factor']['p']
f = criticality['four_factor']['f']
k_infinity = criticality['four_factor']['k_infinity']

# Access six-factor (k-effective)
k_eff = criticality['six_factor']['k_effective']
P_fast = criticality['six_factor']['P_fast']
P_thermal = criticality['six_factor']['P_thermal']

# Reactivity
rho_pcm = criticality['reactivity']['pcm']
rho_dollars = criticality['reactivity']['dollars']

print(f"k_infinity = {k_infinity:.4f}")
print(f"k_effective = {k_eff:.4f}")
print(f"Reactivity = {rho_pcm:.1f} pcm = {rho_dollars:.2f} $")
```

### Neutron Flux Calculations

```python
# Get flux data
flux_data = reactor.calculate_neutron_flux()

print(f"Average flux: {flux_data['average_flux_n_cm2_s']:.3e} n/cm²/s")
print(f"Peak flux: {flux_data['peak_flux_n_cm2_s']:.3e} n/cm²/s")

# Calculate 2D flux distribution
r, z, flux_2d = reactor.neutronics.calculate_flux_distribution(
    r_points=50,
    z_points=100
)
```

### Thermal-Hydraulics Analysis

```python
# Get thermal-hydraulic data
thermal = reactor.calculate_thermal_hydraulics()

# Bulk conditions
print(f"Inlet temperature: {thermal['bulk_conditions']['inlet_temp_K']:.1f} K")
print(f"Outlet temperature: {thermal['bulk_conditions']['outlet_temp_K']:.1f} K")
print(f"Coolant velocity: {thermal['bulk_conditions']['coolant_velocity_m_s']:.2f} m/s")

# Safety margins
print(f"Minimum DNBR: {thermal['safety_margins']['minimum_dnbr']:.2f}")
print(f"Fuel melt margin: {thermal['temperatures']['fuel_melt_margin_K']:.1f} K")

# Pressure drop
print(f"Core pressure drop: {thermal['pressure_drop']['total']:.1f} kPa")
```

### Export Results to JSON

```python
# Export full analysis to JSON
reactor.to_json("reactor_analysis.json")

# Or get JSON string
json_str = reactor.to_json()
```

## Running the Example Script

```bash
# Run basic simulation
python examples/run_simulation.py

# Run with custom parameters
python examples/run_simulation.py --power 2800 --enrichment 4.5

# Run parametric study
python examples/run_simulation.py --study parametric

# Run all analyses
python examples/run_simulation.py --study all

# Export to JSON
python examples/run_simulation.py --output results.json
```

## Project Structure

```
nuclear_core/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── nuclear_core/             # Main package
│   ├── __init__.py          # Package exports
│   ├── constants.py         # Physical constants and nuclear data
│   ├── materials.py         # Fuel, moderator, cladding properties
│   ├── geometry.py          # Core and fuel pin geometry
│   ├── neutronics.py        # Neutron physics calculations
│   ├── thermal.py           # Thermal-hydraulic calculations
│   ├── reactor.py           # Main reactor model
│   └── utils.py             # Utility functions
├── tests/                    # Unit tests
│   ├── test_constants.py
│   ├── test_materials.py
│   ├── test_geometry.py
│   ├── test_neutronics.py
│   ├── test_thermal.py
│   └── test_reactor.py
└── examples/
    └── run_simulation.py     # Example simulation script
```

## Module Descriptions

### constants.py
Contains fundamental physical constants (Avogadro's number, neutron mass, etc.), nuclear cross-section data (U-235, U-238, H-1, O-16), delayed neutron data, and fission product data.

### materials.py
Defines material classes:
- `UO2Fuel`: Uranium dioxide fuel with configurable enrichment (3-5%)
- `LightWaterModerator`: Light water moderator/coolant properties
- `ZircaloyCladding`: Zircaloy-4 cladding properties
- `FuelAssembly`: Complete 17×17 fuel assembly

### geometry.py
Defines geometric specifications:
- `FuelPin`: Single fuel pin dimensions
- `CoreGeometry`: Full reactor core configuration (193 assemblies)
- `ReflectorGeometry`: Water reflector

### neutronics.py
Implements neutron physics:
- `CriticalityCalculator`: Four-factor and six-factor formula calculations
- `NeutronicsModel`: Flux distributions and poisoning calculations
- `TwoGroupDiffusion`: Two-group diffusion theory

### thermal.py
Implements thermal-hydraulics:
- `CoolantProperties`: Water thermophysical properties
- `FuelProperties`: UO2 thermal properties
- `ThermalHydraulics`: Temperature distributions, DNBR, pressure drop

### reactor.py
Main integration:
- `PWRCore`: Complete reactor model integrating all physics
- `create_pwr_core()`: Factory function for easy reactor creation

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_neutronics.py -v

# Run with coverage
python -m pytest tests/ --cov=nuclear_core
```

## Physical Assumptions and Limitations

This model makes several simplifying assumptions:

1. **Homogeneous Core**: Cross-sections are homogenized over the lattice cell
2. **Fundamental Mode**: Flux distributions use fundamental mode approximation
3. **Steady State**: No time-dependent kinetics (except for equilibrium poisoning)
4. **No Burnup**: Fresh fuel only - no depletion calculations
5. **Simplified Correlations**: Thermal-hydraulic properties use simplified correlations

For production reactor analysis, validated lattice physics codes (e.g., CASMO, PARAGON) and core simulators (e.g., SIMULATE, ANC) should be used.

## Typical PWR Parameters (Reference)

| Parameter | Value | Unit |
|-----------|-------|------|
| Thermal Power | 3000 | MW |
| Number of Assemblies | 193 | - |
| Fuel Pins per Assembly | 264 | - |
| Enrichment | 3-5 | wt% U-235 |
| Core Height | 3.66 | m |
| Coolant Inlet Temp | 292 | °C |
| Coolant Outlet Temp | 325 | °C |
| System Pressure | 15.5 | MPa |
| Mass Flow Rate | 17400 | kg/s |

## References

1. Duderstadt, J.J. and Hamilton, L.J., "Nuclear Reactor Analysis", John Wiley & Sons, 1976
2. Lamarsh, J.R. and Baratta, A.J., "Introduction to Nuclear Engineering", 3rd Ed., Prentice Hall, 2001
3. Todreas, N.E. and Kazimi, M.S., "Nuclear Systems I: Thermal Hydraulic Fundamentals", 2nd Ed., CRC Press, 2011
4. IAEA-TECDOC-1496, "Thermophysical Properties Database of Materials for Light Water Reactors and Heavy Water Reactors", 2006

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Disclaimer

This model is for educational and research purposes only. It should not be used for actual reactor design or safety analysis without proper validation against experimental data and benchmarks.
