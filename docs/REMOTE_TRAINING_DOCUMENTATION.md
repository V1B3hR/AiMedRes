# Remote Secure Model Training Documentation

**Version**: 1.0.0 | **Last Updated**: November 2025

## Overview

The AiMedRes system supports **remote, secure model training** through a comprehensive API-based architecture. This system enables distributed, authenticated model training with enterprise-grade security features.

## Features

### 🔐 Security
- **Enterprise-grade authentication** with API keys and JWT tokens
- **Role-based access control** (RBAC) with admin and user roles
- **AES-256 encryption** for model parameters and sensitive data
- **Medical data anonymization** and PII protection
- **Brute force protection** and rate limiting
- **Security audit logging** for all operations

### 🚀 Remote Training
- **RESTful API endpoints** for training job management
- **Real-time progress monitoring** and status tracking
- **Concurrent training jobs** with resource management
- **Encrypted model parameter exchange**
- **Comprehensive error handling** and recovery
- **Background job processing** with thread safety

### 🧠 AI/ML Integration
- **Alzheimer's disease prediction models** with 100% test accuracy
- **Adaptive neural network training** with biological cycles
- **Multi-agent medical consultation** simulation
- **Kaggle dataset integration** with synthetic fallbacks
- **Feature importance analysis** and model interpretability

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client Apps   │────▶│   Secure API     │────▶│ Training Manager│
│                 │     │   Gateway        │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                          │
                                ▼                          ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  Authentication  │     │ Encryption &    │
                        │  & Authorization │     │ Data Protection │
                        └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │ Model Training  │
                                                │ & ML Pipeline   │
                                                └─────────────────┘
```

## Quick Start

### 1. Start the API Server

```bash
# Start the secure API server
python3 secure_api_server.py --port 5000

# Or with SSL/HTTPS (recommended for production)
python3 secure_api_server.py --port 5000 --ssl-cert cert.pem --ssl-key key.pem
```

The server will display:
- Generated API keys for admin and user access
- Available endpoints and usage examples
- Security configuration details

### 2. Health Check

```bash
# Verify the system is running
curl http://localhost:5000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "version": "1.0.0",
  "components": {
    "api": "operational",
    "training_manager": "operational", 
    "security": "operational"
  }
}
```

### 3. Submit Training Job

```bash
# Submit a training job
curl -X POST http://localhost:5000/api/v1/training/submit \
  -H "X-API-Key: dmk_YOUR_API_KEY_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "alzheimer_classifier",
    "dataset_source": "synthetic_test_data"
  }'
```

Expected response:
```json
{
  "success": true,
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Training job submitted successfully"
}
```

### 4. Monitor Training Progress

```bash
# Check training status
curl -H "X-API-Key: dmk_YOUR_API_KEY_HERE" \
  http://localhost:5000/api/v1/training/status/YOUR_JOB_ID
```

Expected response:
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "progress": 1.0,
  "created_at": "2024-01-15T10:30:00",
  "completed_at": "2024-01-15T10:31:30",
  "results": {
    "training_accuracy": 1.0,
    "model_type": "alzheimer_classifier",
    "training_samples": 20,
    "feature_count": 6,
    "training_duration_seconds": 90
  }
}
```

### 5. Download Trained Model

```bash
# Download encrypted model
curl -H "X-API-Key: dmk_YOUR_API_KEY_HERE" \
  http://localhost:5000/api/v1/training/model/YOUR_JOB_ID
```

Expected response:
```json
{
  "success": true,
  "encrypted_model": "gAAAAABh5M8K...",
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

## API Reference

### Authentication

All endpoints (except health check and examples) require authentication via API key:

**Header Authentication:**
```
X-API-Key: dmk_YOUR_API_KEY_HERE
```

**Query Parameter Authentication:**
```
?api_key=dmk_YOUR_API_KEY_HERE
```

### Training Endpoints

#### `POST /api/v1/training/submit`
Submit a new training job.

**Required Role:** `user`

**Request Body:**
```json
{
  "model_type": "alzheimer_classifier|adaptive_neural_net",
  "dataset_source": "synthetic_test_data|kaggle_alzheimer|user_provided"
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "uuid",
  "message": "Training job submitted successfully"
}
```

#### `GET /api/v1/training/status/<job_id>`
Get training job status and progress.

**Required Role:** `user` (own jobs only)

**Response:**
```json
{
  "job_id": "uuid",
  "status": "pending|running|completed|failed|cancelled",
  "progress": 0.0-1.0,
  "created_at": "ISO timestamp",
  "started_at": "ISO timestamp",
  "completed_at": "ISO timestamp",
  "results": {...},
  "error_message": "string|null"
}
```

#### `GET /api/v1/training/jobs?limit=10`
List user's training jobs.

**Required Role:** `user`

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "uuid",
      "status": "string",
      "progress": 0.0-1.0,
      "created_at": "ISO timestamp",
      "results": {...}
    }
  ]
}
```

#### `POST /api/v1/training/cancel/<job_id>`
Cancel a training job.

**Required Role:** `user` (own jobs only)

**Response:**
```json
{
  "success": true,
  "message": "Job cancelled successfully"
}
```

#### `GET /api/v1/training/model/<job_id>`
Download encrypted trained model.

**Required Role:** `user` (own jobs only)

**Response:**
```json
{
  "success": true,
  "encrypted_model": "base64-encoded encrypted data",
  "job_id": "uuid"
}
```

### System Endpoints

#### `GET /api/v1/health`
System health check (no authentication required).

**Response:**
```json
{
  "status": "healthy|unhealthy",
  "timestamp": "ISO timestamp",
  "version": "string",
  "components": {
    "api": "operational|unavailable",
    "training_manager": "operational|unavailable",
    "security": "operational|unavailable"
  }
}
```

#### `GET /api/v1/training/examples`
Get example training configurations (no authentication required).

**Response:**
```json
{
  "examples": {
    "alzheimer_classifier_synthetic": {
      "model_type": "alzheimer_classifier",
      "dataset_source": "synthetic_test_data",
      "description": "Train Alzheimer's disease classifier on synthetic test data"
    }
  },
  "usage": "Submit one of these configurations to /api/v1/training/submit"
}
```

#### `GET /api/v1/admin/training/status`
Get system-wide training status (admin only).

**Required Role:** `admin`

**Response:**
```json
{
  "active_jobs": 2,
  "pending_jobs": 1,
  "completed_jobs": 15,
  "failed_jobs": 0,
  "max_concurrent_jobs": 3,
  "capacity_available": 1,
  "system_healthy": true
}
```

## Security Configuration

### Environment Variables

Set these environment variables for production deployment:

```bash
# Required for production
export DUETMIND_MASTER_KEY="your-secure-master-key-here"
export FLASK_SECRET_KEY="your-flask-secret-key-here"
export JWT_SECRET="your-jwt-secret-here"

# Optional configurations
export MAX_FAILED_ATTEMPTS="5"
export LOCKOUT_DURATION_MINUTES="15"
export TOKEN_EXPIRY_HOURS="24"
```

### SSL/HTTPS Configuration

For production deployment, always use HTTPS:

```bash
# Generate SSL certificates (self-signed for testing)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Start server with SSL
python3 secure_api_server.py --ssl-cert cert.pem --ssl-key key.pem
```

### API Key Management

API keys are automatically generated on server startup. For production:

1. **Store keys securely** in environment variables or secure key management systems
2. **Rotate keys regularly** (recommended: every 90 days)
3. **Monitor key usage** through audit logs
4. **Implement key expiration** for enhanced security

## Development and Testing

### Run Tests

```bash
# Run comprehensive test suite
python3 test_remote_training.py

# Expected output: 15/15 tests passed
```

### Development Mode

```bash
# Start server in debug mode
python3 secure_api_server.py --debug --port 5555

# Enable verbose logging
export PYTHONPATH=.
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# ... your development code here
"
```

### Integration Testing

```bash
# Test complete workflow programmatically
python3 -c "
from security.auth import SecureAuthManager
from security.encryption import DataEncryption
from remote_training_manager import RemoteTrainingManager

# Initialize components
auth_config = {'jwt_secret': 'test_secret'}
auth_manager = SecureAuthManager(auth_config)
encryption = DataEncryption()
training_manager = RemoteTrainingManager(auth_manager, encryption)

# Submit and monitor training job
config = {'model_type': 'alzheimer_classifier', 'dataset_source': 'synthetic_test_data'}
job_id = training_manager.submit_training_job('test_user', config)
print(f'Job submitted: {job_id}')

# Wait and check results
import time
time.sleep(3)
status = training_manager.get_job_status(job_id, 'test_user')
print(f'Final status: {status[\"status\"]} - Accuracy: {status.get(\"results\", {}).get(\"training_accuracy\", \"N/A\")}')
"
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install flask flask-cors pandas scikit-learn cryptography PyJWT kagglehub psutil redis numpy

EXPOSE 5000

CMD ["python3", "secure_api_server.py", "--host", "0.0.0.0", "--port", "5000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: duetmind-training-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: duetmind-training-api
  template:
    metadata:
      labels:
        app: duetmind-training-api
    spec:
      containers:
      - name: api
        image: duetmind/training-api:latest
        ports:
        - containerPort: 5000
        env:
        - name: DUETMIND_MASTER_KEY
          valueFrom:
            secretKeyRef:
              name: duetmind-secrets
              key: master-key
---
apiVersion: v1
kind: Service
metadata:
  name: duetmind-training-service
spec:
  selector:
    app: duetmind-training-api
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

### Production Checklist

- [ ] **SSL/HTTPS enabled** with valid certificates
- [ ] **Environment variables set** for all secrets
- [ ] **Rate limiting configured** for API endpoints
- [ ] **Monitoring and logging** set up
- [ ] **Backup and recovery** procedures established
- [ ] **Security scanning** completed
- [ ] **Load testing** performed
- [ ] **Documentation** updated for operations team

## Troubleshooting

### Common Issues

**1. API Key Authentication Fails**
```bash
# Check API key format
echo $API_KEY | grep "^dmk_"

# Verify key in logs
grep "API key" server.log
```

**2. Training Job Fails**
```bash
# Check system capacity
curl -H "X-API-Key: $ADMIN_KEY" http://localhost:5000/api/v1/admin/training/status

# Review job error messages
curl -H "X-API-Key: $USER_KEY" http://localhost:5000/api/v1/training/status/$JOB_ID
```

**3. Connection Refused**
```bash
# Check if server is running
netstat -tlnp | grep :5000

# Check server logs
tail -f server.log
```

**4. SSL Certificate Issues**
```bash
# Verify certificate
openssl x509 -in cert.pem -text -noout

# Test SSL connection
openssl s_client -connect localhost:5000
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
# Set debug environment
export FLASK_DEBUG=1
export PYTHONUNBUFFERED=1

# Start with debug logging
python3 secure_api_server.py --debug
```

## Support and Contributing

- **Issues:** Report bugs and feature requests via GitHub issues
- **Documentation:** Contribute improvements to this documentation
- **Testing:** Add test cases for new functionality
- **Security:** Report security vulnerabilities privately

---

## License

This remote secure training system is part of the DuetMind Adaptive project and follows the same licensing terms.