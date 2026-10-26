# Security Policy

**Version**: 1.0.0 | **Last Updated**: October 2026

> This policy describes how to report suspected vulnerabilities. It does **not** certify the software, its deployments, or its data-processing controls as secure, compliant, or clinically ready. Current audit scope and open risks are recorded in [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

## Active deployment advisory — eMudhra / Node.js (25 October 2026)

The repository has a source-level response to Render's [eMudhra advisory](https://render.com/docs/advisory-emudhra): `.node-version` pins Node.js `24.22.1`, Python environments require `certifi>=2026.10.19`, and all maintained Dockerfiles explicitly refresh `ca-certificates` during a rebuild. These changes do **not** update or verify an already deployed service.

- **PHP2Shell:** the AiMedRes codebase and maintained Dockerfiles contain no PHP runtime, Composer manifest or PHP base image. The optional third-party `adminer:latest` service was removed from `mlops/infra/docker-compose.yml`, so new local MLOps stacks do not start a PHP service by default. Any already running Adminer container, separately managed PHP service, or custom compose override still needs the procedure in Render's [PHP2Shell advisory](https://render.com/docs/advisory-php2shell): replace it with a vendor image verified to contain a patched PHP branch, then redeploy and confirm its runtime version.
- **Render native services:** set `NODE_VERSION=24.22.1` (or use the committed `.node-version`), then trigger a new build/deploy for every service that builds or runs Node.js. Render notes that native runtimes can use Node during builds even when the application itself is Python.
- **Docker services:** build with a fresh base and package layers, for example `docker build --pull --no-cache -t aimedres:security-refresh .`, then redeploy the new image/reference. Compose-managed third-party images must also be refreshed with their vendor-supported pull/rebuild process.
- **Verification after redeploy:** run the application's own HTTPS client from the deployed service against the two certificate-rejection endpoints listed in the Render advisory. Both requests must fail certificate validation. Do not treat a local workstation `curl` result as evidence for the deployed service.

Record the deployed image digest, runtime Node version, deploy ID/time and verification result with the release. The deployment owner must perform those actions; this repository cannot inspect or redeploy Render services.

## 🔒 Reporting Security Vulnerabilities

**We take security seriously.** If you discover a security vulnerability in AiMedRes, please report it responsibly.

### ⚠️ IMPORTANT: Do NOT Create Public Issues for Security Vulnerabilities

Security vulnerabilities should **NEVER** be reported through public GitHub issues, discussions, or pull requests.

### Reporting Process

#### Option 1: GitHub Private Vulnerability Reporting (Recommended)

1. Go to the [Security tab](https://github.com/V1B3hR/AiMedRes/security) of this repository
2. Click "Report a vulnerability"
3. Fill out the vulnerability report form with detailed information
4. Submit the report

GitHub will privately notify the maintainers, and we can discuss the issue securely.

#### Option 2: Email Reporting

If you prefer or cannot use GitHub's private reporting:

**Security Contact:** Use a verified maintainer contact published outside public issue content. If no private channel is available, do not disclose vulnerability details in a public issue; request that the maintainer publishes a private reporting channel without including technical details.

### What to Include in Your Report

A good security report includes:

1. **Description**: Clear description of the vulnerability
2. **Impact**: Potential impact if exploited
3. **Steps to Reproduce**: Detailed steps to reproduce the vulnerability
4. **Proof of Concept**: Code or commands demonstrating the issue (if applicable)
5. **Suggested Fix**: Proposed solution (if you have one)
6. **Affected Versions**: Which versions are affected
7. **Environment**: System details (OS, Python version, dependencies)

### Example Report Template

```
**Summary**: Brief description of the vulnerability

**Severity**: Critical / High / Medium / Low

**Affected Component**: Which part of AiMedRes is affected

**Vulnerability Type**: (e.g., SQL Injection, XSS, Path Traversal, etc.)

**Description**: 
Detailed description of the vulnerability and how it can be exploited.

**Impact**:
What an attacker could achieve by exploiting this vulnerability.

**Steps to Reproduce**:
1. Step 1
2. Step 2
3. Step 3

**Proof of Concept**:
[Code or commands demonstrating the issue]

**Affected Versions**:
- Version X.Y.Z
- Version A.B.C

**Suggested Mitigation**:
[Your suggestions, if any]

**Environment**:
- OS: Ubuntu 20.04
- Python: 3.10.0
- AiMedRes version: 1.0.0
```

## Response Timeline

We are committed to responding to security reports promptly:

- **Initial Response**: Within 48 hours of report submission
- **Status Update**: Within 5 business days with initial assessment
- **Resolution Target**: 
  - Critical vulnerabilities: 7 days
  - High severity: 14 days
  - Medium severity: 30 days
  - Low severity: 90 days

**Note**: These are targets. Complex vulnerabilities may require more time.

## Disclosure Policy

### Coordinated Disclosure

We follow a **coordinated disclosure** approach:

1. **Private Discussion**: We'll work with you privately to understand and fix the issue
2. **Fix Development**: We'll develop and test a fix
3. **Release**: We'll release the fix in a security update
4. **Public Disclosure**: After the fix is released, we'll:
   - Publish a security advisory
   - Credit you (if you wish) in the advisory
   - Document the vulnerability in release notes

### Disclosure Timeline

- **90 Days**: We aim to fix and release patches within 90 days
- **Public Disclosure**: After fix is released, or after 90 days (whichever comes first)
- **Early Disclosure**: If the vulnerability is being actively exploited, we may disclose earlier

### Embargo Period

Please allow us reasonable time to:
- Investigate the issue
- Develop a fix
- Test the fix thoroughly
- Release the patch
- Notify affected users

**We request a 90-day embargo** before public disclosure.

## Scope

### In Scope

Security vulnerabilities in:

- ✅ Core AiMedRes codebase (`src/aimedres/`)
- ✅ API endpoints and web interfaces
- ✅ Authentication and authorization systems
- ✅ Data encryption and privacy features
- ✅ Input validation and sanitization
- ✅ PHI/PII handling and de-identification
- ✅ Dependencies with known vulnerabilities
- ✅ Configuration security issues

### Out of Scope

The following are generally **NOT** considered security vulnerabilities:

- ❌ Vulnerabilities in dependencies (report to the dependency maintainers)
- ❌ Issues requiring physical access to the server
- ❌ Social engineering attacks
- ❌ Denial of Service (DoS) attacks requiring excessive resources
- ❌ Issues in third-party integrations (unless they affect AiMedRes security)
- ❌ Theoretical vulnerabilities without proof of concept
- ❌ Known issues already in our issue tracker

**Note**: If unsure, report it anyway. We'll help determine if it's in scope.

## Security Measures

### Current Security Features

AiMedRes implements multiple security layers:

#### Authentication & Authorization
- Secure API key management with PBKDF2 hashing
- Role-based access control (RBAC)
- JWT-based session management
- Rate limiting to prevent abuse

#### Data Protection
- AES-256-GCM encryption for sensitive data at rest
- TLS/HTTPS for data in transit
- Automatic PHI/PII de-identification
- Secure data handling procedures

#### Privacy Compliance
- HIPAA compliance features
- GDPR compliance features
- Audit logging for all data access
- Data retention policies

#### Input Validation
- SQL injection prevention
- XSS protection
- Path traversal prevention
- Command injection prevention

#### Monitoring & Logging
- Security event logging
- Anomaly detection
- Failed authentication tracking
- Audit trails for compliance

See [docs/SECURITY_CONFIGURATION.md](docs/SECURITY_CONFIGURATION.md) for detailed information.

## Security Best Practices for Users

### For Developers

1. **Keep Dependencies Updated**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Use Virtual Environments**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Never Commit Secrets**
   - Use environment variables
   - Add secrets to `.gitignore`
   - Use `.env.example` for templates

4. **Follow Secure Coding Practices**
   - Validate all inputs
   - Use parameterized queries
   - Escape output
   - Follow principle of least privilege

### For Deployments

1. **Secure Configuration**
   - Set strong encryption keys
   - Enable HTTPS/TLS
   - Configure firewalls
   - Use secure passwords

2. **Environment Variables**
   ```bash
   export AIMEDRES_MASTER_KEY="secure-random-key"
   export AIMEDRES_JWT_SECRET="secure-jwt-secret"
   ```

3. **Access Control**
   - Limit file permissions
   - Use non-root users
   - Configure network isolation
   - Enable audit logging

4. **Regular Updates**
   - Monitor security advisories
   - Apply security patches promptly
   - Review dependency vulnerabilities

### For Medical/Research Use

1. **PHI Protection**
   - Always de-identify data
   - Use secure data processors
   - Follow HIPAA/GDPR requirements
   - Obtain appropriate approvals

2. **Research Ethics**
   - Get IRB approval
   - Obtain informed consent
   - Maintain data privacy
   - Follow institutional policies

## Known Security Considerations

### MLflow AI Gateway advisory (GHSA-h7x2-h6g9-p789)

The pinned MLflow release range is affected by an SSRF advisory in the optional
AI Gateway feature. No patched MLflow release is currently available. The
provided Compose deployments therefore bind MLflow to `127.0.0.1` only; do not
publish the MLflow port directly to untrusted networks. If remote access is
required, place it behind an authenticated reverse proxy with network egress
controls. Track the upstream advisory and update MLflow as soon as a patched
release is published.

### Medical AI Specific Risks

As medical AI software, AiMedRes has unique security considerations:

1. **Model Poisoning**: Malicious training data could compromise models
2. **Adversarial Examples**: Crafted inputs could fool AI models
3. **Privacy Leakage**: Models might leak training data
4. **Fairness Issues**: Biased data could lead to discriminatory outcomes

**Mitigation**: 
- Validate training data sources
- Use adversarial training techniques
- Implement differential privacy where possible
- Regular bias audits

### Research Use Only

⚠️ **IMPORTANT**: AiMedRes is for **research use only** and is **NOT** approved for clinical diagnosis or treatment.

Security vulnerabilities in research software may have different implications than in production medical devices. However, we still take all security reports seriously.

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| 1.0.x   | :white_check_mark: | Current research-development version |
| < 1.0   | :x:                | Unsupported pre-release versions |

**Note**: As AiMedRes is in active development, we recommend using the latest version from the main branch.

## Security Hall of Fame

We recognize and thank security researchers who responsibly disclose vulnerabilities:

| Researcher | Vulnerability | Disclosed | Severity |
|------------|---------------|-----------|----------|
| *None yet* | -             | -         | -        |

*Your name could be here! Report a valid security vulnerability to be recognized.*

## Security Resources

### Internal Documentation
- [Security Configuration Guide](docs/SECURITY_CONFIGURATION.md)
- [Security Guidelines for Contributors](docs/SECURITY_GUIDELINES.md)
- [Data Handling Procedures](docs/DATA_HANDLING_PROCEDURES.md)

### External Resources
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/)

## Legal

### Responsible Disclosure Agreement

By reporting a security vulnerability, you agree to:

1. **Not Exploit**: Not exploit the vulnerability beyond what's necessary to demonstrate it
2. **Confidentiality**: Keep the vulnerability confidential until we release a fix
3. **Good Faith**: Act in good faith to avoid privacy violations, destruction of data, or interruption of service
4. **Compliance**: Comply with all applicable laws

We agree to:

1. **Acknowledge**: Acknowledge your report promptly
2. **Investigate**: Investigate and validate the report
3. **Fix**: Work diligently to fix the vulnerability
4. **Credit**: Credit you in our security advisory (if you wish)
5. **No Legal Action**: Not pursue legal action for good-faith security research

### Immunity

We will not pursue legal action against researchers who:
- Report vulnerabilities responsibly
- Do not exploit vulnerabilities maliciously
- Act in good faith
- Comply with this policy

## Questions?

If you have questions about this security policy or the reporting process:

1. Check our [Security Configuration Guide](docs/SECURITY_CONFIGURATION.md)
2. Review our [Security Guidelines](docs/SECURITY_GUIDELINES.md)
3. Open a general (non-security) GitHub issue asking for clarification
4. Contact the maintainers through GitHub Discussions

**Remember**: Never discuss specific vulnerabilities publicly until they are fixed and disclosed.

---

**Thank you for helping keep AiMedRes and its users secure!** 🔒

*Last Updated: 2026-10-23*
*Version: 1.0*
