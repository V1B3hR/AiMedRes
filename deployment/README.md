# AiMedRes Deployment Guide

This directory contains all necessary files and documentation for deploying AiMedRes in healthcare environments.

## Directory Structure

```
deployment/
├── README.md                          # This file
├── preparation/                       # Step 1: Preparation and Planning
│   ├── system_requirements.md         # Hardware and software requirements
│   ├── stakeholder_alignment.md       # Stakeholder identification and alignment checklist
│   └── legal_risk_assessment.md       # Legal, regulatory, and risk assessment template
├── technical/                         # Step 2: Technical Environment Setup
│   ├── Dockerfile                     # Production Docker container definition
│   ├── docker-compose.yml             # Complete stack orchestration
│   ├── docker-entrypoint.sh           # Container initialization script
│   ├── .env.healthcare.template       # Environment configuration template
│   ├── system_hardening_guide.md      # Security hardening procedures
│   ├── init-db.sql                    # Database initialization script
│   └── mlflow-requirements.txt        # MLflow dependencies
└── templates/                         # Reusable templates (future use)
```

## Quick Start

### Prerequisites
- Docker 20.10+ and Docker Compose 2.0+
- Review `preparation/system_requirements.md` for complete requirements

### Deployment Steps

#### 1. Complete Preparation Phase (Step 1)
Work through all documents in the `preparation/` directory:

1. **System Requirements**: `preparation/system_requirements.md`
   - Verify hardware and software meet minimum requirements
   - Plan for integration endpoints (EMR/EHR, PACS, etc.)
   - Document compliance requirements

2. **Stakeholder Alignment**: `preparation/stakeholder_alignment.md`
   - Identify all stakeholders
   - Complete engagement activities
   - Obtain necessary approvals

3. **Legal & Risk Assessment**: `preparation/legal_risk_assessment.md`
   - Complete regulatory compliance assessment (HIPAA, GDPR, FDA)
   - Document risk mitigation strategies
   - Obtain legal and compliance sign-offs

#### 2. Set Up Technical Environment (Step 2)

1. **Prepare Configuration**
   ```bash
   cd deployment/technical
   cp .env.healthcare.template .env
   # Edit .env and fill in all required values
   nano .env
   ```

2. **Generate Security Keys**
   ```bash
   # Generate random secure keys for SECRET_KEY, ENCRYPTION_KEY, etc.
   python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(64))"
   python3 -c "import secrets; print('ENCRYPTION_KEY=' + secrets.token_urlsafe(64))"
   python3 -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(64))"
   ```

3. **Review and Customize Docker Configuration**
   - Review `Dockerfile` for any organization-specific requirements
   - Review `docker-compose.yml` for service configuration
   - Adjust resource limits based on your hardware

4. **System Hardening**
   - Follow procedures in `technical/system_hardening_guide.md`
   - Apply OS-level security configurations
   - Configure firewall rules
   - Set up audit logging

5. **Deploy Services**
   ```bash
   # Build and start all services
   docker-compose up -d
   
   # Check service health
   docker-compose ps
   docker-compose logs -f aimedres
   
   # Verify database initialization
   docker-compose exec postgres psql -U aimedres -d aimedres -c "\dt aimedres.*"
   ```

6. **Verify Deployment**
   ```bash
   # Check application health
   curl http://localhost:8002/health
   
   # Check MLflow
   curl http://localhost:5001/health
   
   # Check MinIO
   curl http://localhost:9000/minio/health/live
   ```

## Configuration Files

### Environment Variables (.env)
Copy `.env.healthcare.template` to `.env` and configure:

**Critical Settings:**
- `SECRET_KEY`, `ENCRYPTION_KEY` - Generate strong random values
- `POSTGRES_PASSWORD`, `MINIO_ROOT_PASSWORD`, `REDIS_PASSWORD` - Set secure passwords
- `ENABLE_PHI_SCRUBBING=true` - Always enable in production
- `ENABLE_AUDIT_LOGGING=true` - Required for HIPAA compliance
- `ENABLE_ENCRYPTION_AT_REST=true` - Required for PHI protection

### Docker Compose Customization
Adjust `docker-compose.yml` for your environment:
- Resource limits (CPU, memory)
- Port mappings
- Volume mounts for persistent data
- Network configuration

## Security Considerations

### Pre-Deployment
- [ ] Complete system hardening guide procedures
- [ ] Generate and securely store all encryption keys
- [ ] Configure TLS/SSL certificates
- [ ] Set up firewall rules
- [ ] Enable audit logging
- [ ] Configure backup procedures

### Post-Deployment
- [ ] Verify all services are running with proper permissions
- [ ] Test PHI scrubbing functionality
- [ ] Verify audit logs are being generated
- [ ] Test backup and restore procedures
- [ ] Conduct security scan
- [ ] Review access controls

## Maintenance

### Daily
- Monitor service health and logs
- Review security alerts

### Weekly
- Review audit logs
- Check backup completion

### Monthly
- Apply security updates
- Review and rotate logs
- Test disaster recovery

### Quarterly
- Full security audit
- Update documentation
- Review and update access controls

## Troubleshooting

### Services Won't Start
```bash
# Check logs
docker-compose logs

# Check specific service
docker-compose logs aimedres
docker-compose logs postgres

# Restart services
docker-compose restart
```

### Database Connection Issues
```bash
# Verify database is running
docker-compose ps postgres

# Test database connection
docker-compose exec postgres psql -U aimedres -d aimedres -c "SELECT 1;"
```

### Permission Errors
```bash
# Fix volume permissions
docker-compose down
sudo chown -R 1000:1000 ./volumes/
docker-compose up -d
```

## Support

For issues or questions:
1. Review documentation in this directory
2. Check application logs: `docker-compose logs -f aimedres`
3. Review the main project README.md
4. Contact your technical support team

## Compliance & Audit

This deployment includes:
- ✅ Audit logging for all data access
- ✅ PHI scrubbing capabilities
- ✅ Encryption at rest and in transit
- ✅ Role-based access control (RBAC)
- ✅ Comprehensive security logging
- ✅ Backup and disaster recovery support

Ensure all compliance documentation is completed before production use.

## References

- Main Project: `/README.md`
- Security Documentation: `/SECURITY.md`
- Healthcare Deployment Plan: `/healthcaredeploymentplan.md`
- API Documentation: `/docs/API_REFERENCE.md`

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-16  
**Next Review**: Before each deployment
