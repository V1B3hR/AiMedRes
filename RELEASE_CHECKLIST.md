# AiMedRes Pre-Release Checklist

This checklist ensures the repository is ready for external release to clinical/academic partners.

## 📋 Documentation

- [x] Consolidate documentation under `docs/` directory
- [x] Create documentation index (`docs/INDEX.md`)
- [x] Update README.md with correct documentation links
- [x] Archive historical summaries in `docs/archive/`
- [ ] Review and update all API documentation
- [ ] Ensure all guides are up-to-date and accurate
- [ ] Verify all cross-references and links work
- [ ] Add examples for all major features

## 🧹 Code Cleanup

- [x] Remove legacy `.shim` compatibility wrapper files
- [x] Remove unused imports from core modules
- [ ] Remove debug print statements
- [ ] Clean up hardcoded paths
- [ ] Remove excessive logging statements
- [ ] Standardize code formatting (run black/autopep8)
- [ ] Fix flake8 warnings
- [ ] Address mypy type checking issues

## 🧪 Testing

- [ ] Install all dependencies successfully from requirements
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Training scripts run end-to-end from fresh clone
- [ ] Examples run without errors
- [ ] CLI commands work as documented
- [ ] No import errors in any module

## 🔒 Security & Compliance

- [ ] Review HIPAA compliance documentation
- [ ] Verify IRB procedures are documented
- [ ] Ensure data privacy guidelines are clear
- [ ] Review and update security documentation
- [ ] Verify no secrets or credentials in code
- [ ] Check for exposed API keys or tokens
- [ ] Audit logging is properly configured
- [ ] PHI scrubber is enabled and tested in CI
- [ ] Vulnerability disclosure process is documented (SECURITY.md)
- [ ] Human-in-loop gating is enforced for high-risk recommendations

## 🏥 Clinical Use Classification & Disclaimers (P0-2)

- [ ] **Clinical Use Classification**: Clearly documented as "Research Only / Not a Diagnostic Device"
- [ ] **README.md**: Contains prominent clinical use warnings and disclaimers
- [ ] **UI Templates**: All UI templates include research-only disclaimers
- [ ] **About Page**: Created with comprehensive disclaimers and classification
- [ ] **Documentation**: All relevant docs include appropriate clinical use warnings
- [ ] **API Responses**: API includes disclaimer fields in responses
- [ ] **CLI Output**: CLI tools include disclaimer text in help and output
- [ ] **Legal Signoff**: Clinical use classification reviewed with legal/clinical lead
- [ ] **FDA Classification**: Documented that software is NOT an FDA-regulated medical device
- [ ] **Professional Judgment**: Disclaimers emphasize need for licensed professional oversight

## 📊 Reproducibility

- [ ] Seed values are documented and configurable
- [ ] Configuration files work as expected
- [ ] Sample data is available or documented
- [ ] Training results are reproducible
- [ ] Environment setup is clearly documented
- [ ] Dependencies are pinned to specific versions

## 🚀 Deployment

- [ ] Setup instructions are complete
- [ ] Installation process is tested on clean environment
- [ ] Docker/container setup works (if applicable)
- [ ] MLOps pipelines are documented
- [ ] Monitoring and logging are configured
- [ ] Performance benchmarks are documented

## 📝 Legal & Administrative

- [ ] License is clearly stated (GPL-3.0)
- [ ] Copyright notices are accurate
- [ ] Third-party licenses are acknowledged
- [ ] CONTRIBUTING.md is complete
- [ ] Code of conduct is present (if needed)
- [ ] Citation information is provided
- [ ] Legal signoff tracked (see docs/LEGAL_SIGNOFF.md)

## 🎯 User Experience

- [ ] Quickstart guide is clear and works
- [ ] Error messages are helpful
- [ ] Progress indicators work correctly
- [ ] Output formats are documented
- [ ] CLI help text is comprehensive
- [ ] Configuration options are documented

## 📦 Package & Distribution

- [ ] Package version is set correctly
- [ ] setup.py/pyproject.toml is complete
- [ ] Package installs via pip
- [ ] Entry points work correctly
- [ ] Package metadata is accurate
- [ ] Dependencies are correctly specified

## ✅ Final Verification

- [ ] CHANGELOG.md is up to date
- [ ] README.md is comprehensive
- [ ] All documentation links are valid
- [ ] No TODO or FIXME comments in production code
- [ ] Code is properly commented
- [ ] No dead code or unused files
- [ ] Repository structure is clean and logical
- [ ] Git history is clean (no sensitive data)

## 🚢 Release Preparation

- [ ] Create release notes
- [ ] Tag release version
- [ ] Create GitHub release
- [ ] Announce to stakeholders
- [ ] Prepare training materials
- [ ] Set up support channels

---

## Notes

Use this checklist systematically before releasing to external partners. Each item should be verified and checked off before proceeding with the release.

**Target Release Date**: TBD  
**Release Version**: 1.0.0  
**Release Manager**: TBD
