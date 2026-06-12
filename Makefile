.PHONY: install dev test lint build up down deploy

install:
	uv pip install --system -e .[dev]

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check app/ tests/
	ruff format --check app/ tests/

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

deploy:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
