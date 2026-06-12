# Project Name

FastAPI-backend med Docker och automatisk deploy till VPS.

## Lokal utveckling

```bash
cp .env.example .env
make install
make dev
```

API: http://localhost:8000  
Docs: http://localhost:8000/docs

## Tester

```bash
make test
```

## Deploy

Push till `main` triggar automatisk deploy via GitHub Actions.

### GitHub Secrets
- `VPS_HOST` — IP till servern
- `VPS_USER` — SSH-användare (root eller pappa)
- `VPS_SSH_KEY` — privat SSH-nyckel
