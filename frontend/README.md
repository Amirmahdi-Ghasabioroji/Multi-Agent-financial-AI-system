# MAFAS Research Workstation

Dark, responsive Next.js dashboard for the Multi-Agent Financial Analysis
System. It provides full-pipeline and specialist workspaces, run history, corpus
controls, live SSE telemetry and printable reports.

## Configuration

Copy `.env.example` to `.env.local` and set the browser-accessible API root:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

Demo/live labels reflect the requested or reported mode. Always verify source
freshness and citation dates.

## Development and verification

```bash
npm install
npm run dev
npm run lint
npm run typecheck
npm test
npm run build
```

When Node is unavailable on the host:

```bash
docker run --rm -v "$PWD:/app" -w /app node:22-alpine npm test
docker run --rm -v "$PWD:/app" -w /app node:22-alpine npm run build
```

## Production image

`NEXT_PUBLIC_API_URL` is embedded in client assets at build time:

```bash
docker build \
  --build-arg NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1 \
  -t mafas-frontend .
docker run --rm -p 3000:3000 mafas-frontend
```

All outputs are research simulations, not financial advice or trading
instructions.
