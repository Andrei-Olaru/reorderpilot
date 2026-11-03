# One image for both services (api + ui); docker-compose.yml picks the command.
# Data (DATA_DIR) and the trained model (artifacts/) are NOT baked in: they are mounted as volumes,
# because both are gitignored and the raw dataset must not be redistributed.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REORDERPILOT_DATA_DIR=/data

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Editable install keeps config.PROJECT_ROOT = /app, so artifacts/ and eval/ resolve as in a local clone.
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps -e .

COPY app ./app
COPY eval ./eval
RUN mkdir -p artifacts /data

EXPOSE 8000 8501
CMD ["uvicorn", "reorderpilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
