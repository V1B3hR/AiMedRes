# AiMedRes Frontend

## Overview

This directory contains the React + TypeScript frontend for the AiMedRes Clinical Decision Support System.

## Features

### Clinical Decision Support
- **Login Page**: Authentication with username/password
- **Case List View**: Browse and filter clinical cases
- **Case Detail View**: Detailed case information with AI predictions
- **Explainability Panel**: Feature attributions and uncertainty visualization
- **Human-in-Loop Controls**: Clinician approval workflow with rationale

### 3D Visualization & Medical Imaging (NEW)
- **3D Brain Visualizer**: Interactive anatomical mapping with Three.js
  - Real-time disease progression visualization
  - AI explainability overlays
  - Region-specific analysis
  - Treatment impact simulation
- **DICOM Viewer**: Professional medical image viewing with Cornerstone.js
  - Windowing and leveling controls
  - Zoom and pan functionality
  - Multi-series support
  - PACS integration ready

### MLOps & Infrastructure (NEW)
- **Canary Monitoring Dashboard**: Real-time deployment monitoring
  - Shadow mode validation
  - Gradual traffic rollout
  - Automated validation tests
  - Rollback capabilities
- **Quantum Key Management Dashboard**: Quantum-safe cryptography management
  - Key rotation monitoring
  - Policy configuration
  - Usage statistics
  - Audit trail visualization

## Technology Stack

### Core
- **React 18**: UI framework
- **TypeScript**: Type-safe development
- **Vite**: Build tool and dev server
- **React Router**: Client-side routing
- **TanStack Query**: Data fetching and caching
- **Axios**: HTTP client

### 3D Rendering & Medical Imaging
- **Three.js**: 3D graphics library
- **React Three Fiber**: React renderer for Three.js
- **React Three Drei**: Useful helpers for R3F
- **Cornerstone.js**: Medical image viewing
- **DICOM Parser**: DICOM format support

### Testing
- **Cypress**: E2E testing
- **Vitest**: Unit testing

## Quick Start

### Install Dependencies

```bash
cd frontend
npm install
```

### Development Server

```bash
npm run dev
```

Frontend will start on `http://localhost:3000`

### Build for Production

```bash
npm run build
```

### Run Tests

```bash
# Unit tests
npm test

# E2E tests
npm run test:e2e
```

## Project Structure

```
frontend/
├── src/
│   ├── pages/
│   │   ├── Login.tsx                # Login page
│   │   ├── CaseList.tsx             # Case listing
│   │   └── CaseDetail.tsx           # Case detail view
│   ├── components/
│   │   ├── Layout.tsx               # App layout wrapper
│   │   ├── ExplainabilityPanel.tsx  # Feature attributions
│   │   ├── HumanInLoopControls.tsx  # Approval controls
│   │   ├── viewers/
│   │   │   ├── BrainVisualizer.tsx  # 3D brain visualization
│   │   │   └── DICOMViewer.tsx      # DICOM image viewer
│   │   └── dashboards/
│   │       ├── CanaryMonitoringDashboard.tsx    # Canary deployment monitor
│   │       └── QuantumKeyManagementDashboard.tsx # Key management UI
│   ├── api/
│   │   ├── cases.ts             # Cases API client
│   │   ├── viewer.ts            # Viewer API client
│   │   ├── canary.ts            # Canary deployment API
│   │   └── quantum.ts           # Quantum key management API
│   ├── App.tsx                  # Main app component
│   └── main.tsx                 # App entry point
├── cypress/
│   └── e2e/                     # E2E test specs
│       ├── login.cy.ts
│       ├── brain-visualizer.cy.ts
│       ├── dicom-viewer.cy.ts
│       ├── canary-dashboard.cy.ts
│       └── quantum-dashboard.cy.ts
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## API Integration

The frontend connects to the backend API at `http://localhost:8080/api/v1`.

### API Endpoints

#### Authentication
- `POST /api/v1/auth/login` - Authentication

#### Cases
- `GET /api/v1/cases` - List cases
- `GET /api/v1/cases/:id` - Get case detail
- `POST /api/v1/cases/:id/approve` - Approve/reject case

#### Explainability
- `POST /api/v1/explain/attribution` - Get feature attributions
- `POST /api/v1/explain/uncertainty` - Get uncertainty metrics

#### Viewer (NEW)
- `POST /api/viewer/session` - Create viewer session
- `GET /api/viewer/brain/:patientId` - Get brain visualization data
- `GET /api/viewer/dicom/study/:studyId` - Get DICOM study
- `GET /api/viewer/dicom/series/:studyId/:seriesId/stream` - Stream DICOM series

#### Canary Deployment (NEW)
- `GET /api/v1/canary/deployments` - List deployments
- `GET /api/v1/canary/deployments/:id` - Get deployment details
- `GET /api/v1/canary/deployments/:id/metrics` - Get real-time metrics
- `POST /api/v1/canary/deployments/:id/rollback` - Trigger rollback
- `POST /api/v1/canary/deployments/:id/promote` - Promote to stable

#### Quantum Key Management (NEW)
- `GET /api/v1/quantum/keys` - List cryptographic keys
- `GET /api/v1/quantum/keys/:id` - Get key details
- `GET /api/v1/quantum/stats` - Get key manager statistics
- `GET /api/v1/quantum/policy` - Get rotation policy
- `PUT /api/v1/quantum/policy` - Update rotation policy
- `POST /api/v1/quantum/keys/:id/rotate` - Manually rotate key
- `GET /api/v1/quantum/history` - Get rotation history

## Environment Variables

Create `.env.local` file:

```
VITE_API_BASE_URL=http://localhost:8080
VITE_MOCK_API=false
```

## Development with Mock API

For frontend development without backend:

```bash
# In terminal 1: Start mock API server
python src/aimedres/api/mock_server.py

# In terminal 2: Start frontend dev server
npm run dev
```

## Accessibility

The frontend is built with accessibility in mind:
- Semantic HTML
- ARIA labels
- Keyboard navigation
- Screen reader support

## Security

- All API calls include authentication tokens
- PHI data is de-identified
- No sensitive data in localStorage
- HTTPS required in production

## Deployment

### Docker

```bash
docker build -t aimedres-frontend .
docker run -p 3000:3000 aimedres-frontend
```

### Static Hosting

After building, deploy the `dist/` directory to:
- Netlify
- Vercel
- AWS S3 + CloudFront
- Azure Static Web Apps

## Testing Strategy

### Unit Tests

Component tests using Vitest and React Testing Library.

### E2E Tests

```typescript
// cypress/e2e/case-approval.cy.ts
describe('Case Approval Flow', () => {
  it('should allow clinician to approve case', () => {
    cy.visit('/login')
    cy.get('#username').type('clinician-001')
    cy.get('#password').type('password')
    cy.get('button[type=submit]').click()
    
    cy.get('.case-card').first().click()
    cy.get('#approve-button').click()
    cy.get('#rationale').type('Patient presents clear symptoms...')
    cy.get('#submit-approval').click()
    
    cy.contains('Case approved successfully')
  })
})
```

## Performance

- Code splitting for optimal bundle size
- Lazy loading for routes
- React Query for efficient data caching
- Virtual scrolling for large lists

## Browser Support

- Chrome/Edge (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)

## Contributing

1. Create feature branch
2. Make changes
3. Add tests
4. Submit PR

## License

GPL-3.0 (same as parent project)
