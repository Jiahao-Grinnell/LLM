FROM python:3.11-slim

RUN useradd --system --uid 10001 --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ner_extract.py /app/ner_extract.py

USER 10001:10001

ENTRYPOINT ["python", "/app/ner_extract.py"]
