FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts
COPY configs /app/configs
COPY app /app/app
RUN pip install --upgrade pip && pip install -e .[dash]

CMD ["python", "-m", "cryobednet.train", "--config", "configs/mock_cpu.yaml"]
