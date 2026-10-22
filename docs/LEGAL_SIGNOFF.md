# Legal & Licensing Signoff Tracking

## License Standardization: MIT License

### Status: ✅ Reconciled & Harmonized (Phase 0)

**Date**: 2026-10-22  
**Governing License**: **MIT License** (per [`LICENSE`](../LICENSE) file)

---

### 1. Audit Summary & Resolution

A comprehensive governance audit revealed that historical documentation in this repository referenced conflicting licenses (historical notes discussing GPL-3.0, while the root `LICENSE` file, `pyproject.toml`, and Python classifiers declared the MIT License).

To eliminate institutional ambiguity and enable permissive academic and research collaboration, **the repository metadata has been fully reconciled under the MIT License**:
- [x] Verified root `LICENSE` file contains standard MIT License text (`Copyright (c) 2025 V1B3hR`).
- [x] Verified `pyproject.toml` specifies `license = {text = "MIT"}` and `License :: OSI Approved :: MIT License`.
- [x] Harmonized `setup.py` metadata: `LICENSE = "MIT"` and classifier updated.
- [x] Reconciled `README.md` license badge and legal sections from legacy GPL-3.0 text to MIT.
- [x] Aligned web templates (`templates/about.html`, `templates/clinical_dashboard.html`) to MIT terms.

---

### 2. MIT Compliance & Distribution Terms

Under the MIT License:
1. **Permission**: Anyone obtaining a copy of the software and associated documentation may deal in the software without restriction (use, copy, modify, merge, publish, distribute, sublicense, and/or sell).
2. **Attribution**: The above copyright notice and permission notice must be included in all copies or substantial portions of the software.
3. **Disclaimer**: The software is provided "as is", without warranty of any kind, express or implied.

---

### 3. Third-Party Dependency License Audit

All core Python and JavaScript dependencies have been checked for compatibility with MIT distribution:
- **NumPy**: BSD-3-Clause ✅ (Permissive, compatible)
- **Pandas**: BSD-3-Clause ✅ (Permissive, compatible)
- **Scikit-learn**: BSD-3-Clause ✅ (Permissive, compatible)
- **PyTorch**: BSD-style / Apache-2.0 ✅ (Permissive, compatible)
- **Flask**: BSD-3-Clause ✅ (Permissive, compatible)
- **React**: MIT ✅ (Permissive, compatible)

---

### 4. Data Governance vs. Software Licensing

> [!IMPORTANT]
> The MIT License applies exclusively to the open-source software, scripts, configuration, and synthetic benchmark fixtures created by this project.
> It does **NOT** grant distribution rights to third-party clinical or research datasets (e.g., PPMI, Kaggle, ADNI). Such datasets remain governed by their respective institutional Data Use Agreements (DUAs). See [`ParkinsonDatasets/DATA_GOVERNANCE.md`](../ParkinsonDatasets/DATA_GOVERNANCE.md).

---

**Document Version**: 2.0 (Harmonized)  
**Last Updated**: 2026-10-22  
**Maintained by**: AiMedRes Maintainers  
