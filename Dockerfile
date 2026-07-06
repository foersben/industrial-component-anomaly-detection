FROM ubuntu:24.04

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh > install.sh && /bin/bash install.sh && rm install.sh
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml pixi.lock* README.md ./

RUN pixi install -e default

COPY . .

# Assume fastapi usage
CMD ["pixi", "run", "-e", "default", "fastapi", "run", "app/main.py"]
