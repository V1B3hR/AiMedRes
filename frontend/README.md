# AiMedRes Frontend

## Overview

This directory contains the React + TypeScript frontend for the AiMedRes Clinical Decision Support System.

## Features

- **Login Page**: Authentication with username/password
- **Case List View**: Browse and filter clinical cases
- **Case Detail View**: Detailed case information with AI predictions
- **Explainability Panel**: Feature attributions and uncertainty visualization
- **Human-in-Loop Controls**: Clinician approval workflow with rationale

## Technology Stack

- **React 18**: UI framework
- **TypeScript**: Type-safe development
- **Vite**: Build tool and dev server
- **React Router**: Client-side routing
- **TanStack Query**: Data fetching and caching
- **Axios**: HTTP client
- **Cypress**: E2E testing

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
│   │   ├── Login.tsx           # Login page
│   │   ├── CaseList.tsx        # Case listing
│   │   └── CaseDetail.tsx      # Case detail view
│   ├── components/
│   │   ├── Layout.tsx          # App layout wrapper
│   │   ├── ExplainabilityPanel.tsx  # Feature attributions
│   │   └── HumanInLoopControls.tsx  # Approval controls
│   ├── api/
│   │   └── cases.ts            # API client
│   ├── App.tsx                 # Main app component
│   └── main.tsx                # App entry point
├── cypress/
│   └── e2e/                    # E2E test specs
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## API Integration

The frontend connects to the backend API at `http://localhost:8080/api/v1`.

API endpoints used:
- `POST /api/v1/auth/login` - Authentication
- `GET /api/v1/cases` - List cases
- `GET /api/v1/cases/:id` - Get case detail
- `POST /api/v1/cases/:id/approve` - Approve/reject case
- `POST /api/v1/explain/attribution` - Get feature attributions
- `POST /api/v1/explain/uncertainty` - Get uncertainty metrics

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
