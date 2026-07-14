.PHONY: venv install dev test lint build up down deploy demo-video check serve analyze review

# Skapa en isolerad virtuell miljö (rekommenderas på Mac/Linux).
# Aktivera den sedan med:  source .venv/bin/activate
venv:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev]"
	@echo "Klart. Aktivera med: source .venv/bin/activate"

# Installerar i den AKTIVA miljön (venv eller container). Inget --system:
# på Mac kräver systemets Python en venv (annars 'externally-managed' fel).
install:
	pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check app/ tests/ scripts/ analysis/ review/
	ruff format --check app/ tests/ scripts/ analysis/ review/

# Offline analysis (native, not Docker)
analyze:
	python -m analysis.cli

# Review UI (native, not Docker) — open http://localhost:8001
review:
	uvicorn review.main:app --host 0.0.0.0 --port 8001

demo-video:
	python scripts/make_demo_video.py

check:
	python scripts/integration_check.py

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

deploy:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

serve:
	bash scripts/serve.sh
