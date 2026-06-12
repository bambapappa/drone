.PHONY: install dev test lint build up down deploy demo-video check

install:
	uv pip install --system -r pyproject.toml
	uv pip install --system ruff pytest pytest-asyncio

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check app/ tests/ scripts/
	ruff format --check app/ tests/ scripts/

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
