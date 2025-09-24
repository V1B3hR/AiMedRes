# DuetMind Adaptive API Reference

This document provides comprehensive documentation for all API endpoints available in the DuetMind Adaptive system.

## Base URL and Authentication

**Base URL**: `http://localhost:8080` (default)
**Authentication**: API Key required in header: `X-API-Key: your_api_key`

## Available Endpoints

### 1. Health Check
**Endpoint**: `GET /health`  
**Authentication**: None required  
**Description**: System health and status check

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "ISO timestamp",
  "version": "1.0.0",
  "components": {
    "api": "operational",
    "training_manager": "operational", 
    "security": "operational"
  }
}
```

### 2. AI Reasoning
**Endpoint**: `POST /api/v1/reasoning`  
**Authentication**: Required  
**Description**: Submit reasoning tasks to the AI system

**Request Body**:
```json
{
  "query": "Your reasoning query here",
  "context": "Optional context information",
  "model_type": "general|medical|specialized"
}
```

**Response**:
```json
{
  "response": "AI reasoning response",
  "confidence": 0.95,
  "reasoning_chain": ["step1", "step2", "step3"],
  "metadata": {
    "processing_time": 1.23,
    "model_used": "adaptive_v1"
  }
}
```

### 3. Knowledge Search
**Endpoint**: `POST /api/v1/knowledge/search`  
**Authentication**: Required  
**Description**: Search through the knowledge base

**Request Body**:
```json
{
  "query": "Search query",
  "domain": "medical|general",
  "max_results": 10,
  "similarity_threshold": 0.7
}
```

**Response**:
```json
{
  "results": [
    {
      "content": "Knowledge content",
      "source": "Source reference",
      "similarity": 0.95,
      "metadata": {}
    }
  ],
  "total_found": 5,
  "query_time": 0.45
}
```

### 4. Medical Data Processing
**Endpoint**: `POST /api/v1/medical/process`  
**Authentication**: Required  
**Description**: Process medical data with specialized models

**Request Body**:
```json
{
  "data_type": "imaging|text|clinical",
  "data": "Base64 encoded data or JSON structure",
  "analysis_type": "diagnosis|risk_assessment|prediction",
  "patient_id": "anonymized_id"
}
```

**Response**:
```json
{
  "analysis_result": {
    "prediction": "Classification or prediction",
    "confidence": 0.88,
    "risk_factors": ["factor1", "factor2"],
    "recommendations": ["rec1", "rec2"]
  },
  "processing_metadata": {
    "model_version": "medical_v2.1",
    "processing_time": 2.34,
    "compliance_check": "passed"
  }
}
```

### 5. System Metrics (Admin Only)
**Endpoint**: `GET /api/v1/metrics`  
**Authentication**: Admin required  
**Description**: Get system performance and security metrics

**Response**:
```json
{
  "performance": {
    "avg_response_time": 1.23,
    "requests_per_minute": 45,
    "memory_usage": "512MB",
    "cpu_usage": "25%"
  },
  "security": {
    "failed_auth_attempts": 3,
    "active_sessions": 12,
    "last_security_scan": "ISO timestamp"
  },
  "privacy": {
    "data_processed": 150,
    "anonymization_rate": "100%",
    "compliance_status": "GDPR_COMPLIANT"
  }
}
```

### 6. Admin Configuration (Admin Only)
**Endpoint**: `GET /api/v1/admin/config` or `PUT /api/v1/admin/config`  
**Authentication**: Admin required  
**Description**: Get or update system configuration

**GET Response**:
```json
{
  "api_settings": {
    "rate_limit": 100,
    "max_concurrent_requests": 10,
    "timeout": 30
  },
  "security_settings": {
    "encryption_enabled": true,
    "audit_logging": true,
    "session_timeout": 3600
  }
}
```

**PUT Request Body**:
```json
{
  "setting_name": "new_value",
  "rate_limit": 150
}
```

## Error Responses

All endpoints may return these error responses:

### Authentication Error (401)
```json
{
  "error": "Authentication required",
  "request_id": "uuid",
  "timestamp": "ISO timestamp"
}
```

### Authorization Error (403)
```json
{
  "error": "Insufficient permissions",
  "required_role": "admin",
  "request_id": "uuid"
}
```

### Rate Limit Error (429)
```json
{
  "error": "Rate limit exceeded",
  "limit": 100,
  "reset_time": "ISO timestamp",
  "request_id": "uuid"
}
```

### Internal Error (500)
```json
{
  "error": "Internal server error",
  "message": "Error description",
  "request_id": "uuid",
  "timestamp": "ISO timestamp"
}
```

## Rate Limits

- **Default**: 100 requests per minute per API key
- **Admin endpoints**: 50 requests per minute
- **Medical processing**: 20 requests per minute (due to computational requirements)

## Security Notes

1. **API Keys**: Store securely, rotate regularly
2. **HTTPS**: Always use HTTPS in production
3. **Data Privacy**: Medical data is automatically anonymized
4. **Audit Logging**: All API calls are logged for security
5. **Input Validation**: All inputs are validated against XSS/SQL injection

## SDKs and Examples

### Python Example
```python
import requests

headers = {'X-API-Key': 'your_api_key'}
response = requests.post(
    'http://localhost:8080/api/v1/reasoning',
    headers=headers,
    json={'query': 'Analyze this medical case...'}
)
print(response.json())
```

### cURL Example
```bash
curl -X POST http://localhost:8080/api/v1/reasoning \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here"}'
```

## Support

- **Documentation**: Check other guides in the `docs/` folder
- **Issues**: Report via GitHub issues
- **Security**: Follow security guidelines in `SECURITY_GUIDELINES.md`