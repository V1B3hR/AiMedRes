# Configuration Files

This directory contains configuration files for various components:

- `params.yaml` - Main training and model parameters
- `params_imaging.yaml` - Medical imaging pipeline parameters  
- `labyrinth_config.json` - Labyrinth simulation configuration
- `parkinsons_ppmi_study.yml` - Versioned settings for the synthetic PPMI
  Parkinson's research reproducibility benchmark

## Usage

These configuration files are used by the various training and simulation scripts. Modify them to adjust parameters for your specific use case.

## Parkinson's reproducibility benchmark

Run the committed study configuration from the repository root:

```powershell
python scripts/reproduce_ppmi_benchmark.py --config configs/parkinsons_ppmi_study.yml
```

This profile is deliberately restricted to the synthetic, non-identifiable
fixture. It verifies deterministic execution and leakage controls only; it is
not clinical validation or a basis for patient-level decisions. Authorized
researchers may supply an institution-approved controlled dataset with
`--data-path`, subject to the PPMI data-use agreement.
