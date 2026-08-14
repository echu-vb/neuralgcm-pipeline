FROM python:3.12-slim

WORKDIR /app

# system libs some scientific packages need to compile/run
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py main.py ./

ENV OUTPUT_DIR=/app/output
RUN mkdir -p /app/output


CMD ["python", "-u", "main.py"]
