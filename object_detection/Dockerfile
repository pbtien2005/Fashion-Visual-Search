FROM python:3.12.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY requirements-m2.txt ./
RUN pip install --no-cache-dir torch==2.11.0+cpu torchvision==0.26.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-m2.txt

COPY . .

ENV OBJECT_DETECTION_MODEL_PATH=/models/object-detection/best.pt

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
