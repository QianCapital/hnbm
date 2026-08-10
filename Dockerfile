FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml setup.py setup.cfg MANIFEST.in README.md LICENSE ./
COPY hnbm/ hnbm/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

CMD ["python", "-c", "from hnbm import HNBM; print('HNBM ready')"]
