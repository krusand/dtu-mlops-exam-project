FROM python:3.12-slim

RUN pip install uv

COPY README.md README.md
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "uvicorn", "src.exam_project.api:app", "--host", "0.0.0.0", "--port", "8080"]
