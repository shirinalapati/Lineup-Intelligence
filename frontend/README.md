# MLB Lineup Intelligence — Frontend

React + TypeScript + Vite + Tailwind app for the 2026 MLB Lineup Intelligence research platform.

## Develop

```bash
npm install
npm run dev
```

Dev server proxies `/api` → `http://localhost:8200`.

## Build

```bash
npm run build
npm run preview
```

## Routes

- `/` League overview
- `/teams`, `/teams/:abbr`
- `/lineups/:gamePk/:team`
- `/explorer`
- `/players`, `/players/:id`
- `/research`
- `/compare`
