# AiMedRes Mobile Clinical Companion

A React Native / Expo mobile app providing clinicians with on-the-go access to the AiMedRes clinical decision support platform.

## Features

- **Dashboard** — real-time KPI overview (pending cases, high-risk count, model accuracy, alert counts)
- **Cases** — filterable list of clinical cases with risk badges; tap to review full detail including explainability attributions
- **Approve / Reject** — enter a clinical rationale and submit decisions directly from the mobile interface
- **Alerts** — prioritised clinical alert feed with one-tap acknowledgement and severity colour coding

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | [Expo](https://expo.dev) ~50 / React Native 0.73 |
| Language | TypeScript (strict) |
| Navigation | React Navigation 6 (Bottom Tabs + Native Stack) |
| HTTP | Axios |
| Auth | Secure token storage via in-memory cache (upgrade path: Expo SecureStore) |
| Tests | Vitest |

## Getting Started

### Prerequisites

- Node.js ≥ 18
- [Expo CLI](https://docs.expo.dev/get-started/installation/): `npm install -g expo-cli`
- iOS Simulator (macOS) or Android Emulator / physical device with Expo Go

### Install

```bash
cd mobile
npm install
```

### Configure API URL

The app defaults to `http://localhost:8080`. Override via `app.json` extra field or a `.env` file:

```
EXPO_PUBLIC_API_BASE_URL=https://your-backend.example.com
```

### Run

```bash
npm start          # Expo Dev Server
npm run android    # Android emulator
npm run ios        # iOS simulator (macOS only)
```

### Tests

```bash
npm test           # Vitest — run once
npm run test:watch # Vitest — watch mode
```

## Project Structure

```
mobile/
├── App.tsx                        Root component
├── app.json                       Expo config
├── src/
│   ├── api/
│   │   ├── client.ts              Axios instance + token helpers
│   │   ├── cases.ts               Cases & auth API
│   │   ├── alerts.ts              Alerts API
│   │   └── dashboard.ts           Dashboard stats API
│   ├── components/
│   │   ├── AlertItem.tsx          Alert row with severity styling
│   │   ├── CaseCard.tsx           Case summary card
│   │   ├── MetricCard.tsx         Dashboard KPI tile
│   │   └── RiskBadge.tsx          Colour-coded risk level pill
│   ├── navigation/
│   │   └── AppNavigator.tsx       Bottom tab + stack navigator
│   ├── screens/
│   │   ├── DashboardScreen.tsx    KPI overview
│   │   ├── CasesScreen.tsx        Filterable case list
│   │   ├── CaseDetailScreen.tsx   Case detail + decision form
│   │   └── AlertsScreen.tsx       Alert feed with acknowledge
│   ├── types/
│   │   └── index.ts               Shared TypeScript types
│   └── __tests__/
│       ├── setup.ts               Vitest + React Native mocks
│       ├── api.client.test.ts     Token management tests
│       ├── api.cases.test.ts      Cases API tests
│       ├── api.alerts.test.ts     Alerts API tests
│       └── components.test.tsx    Component logic tests
└── vitest.config.ts
```

## Backend API Contract

The app expects the following endpoints on the configured `API_BASE_URL`:

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/auth/login` | POST | Authenticate user |
| `/api/v1/dashboard/stats` | GET | KPI stats |
| `/api/v1/cases` | GET | List cases (`?status=`, `?page=`, `?per_page=`) |
| `/api/v1/cases/:id` | GET | Case detail with explainability |
| `/api/v1/cases/:id/approve` | POST | Approve or reject a case |
| `/api/v1/alerts` | GET | List alerts (`?acknowledged=`, `?page=`, `?per_page=`) |
| `/api/v1/alerts/:id/acknowledge` | POST | Acknowledge an alert |
