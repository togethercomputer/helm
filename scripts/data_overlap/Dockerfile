FROM python:3.12-slim

RUN mkdir -p /script/src

WORKDIR /script/src

COPY . .

RUN pip install --no-cache-dir -r ./requirements-freeze.txt

ENTRYPOINT ["python", "compute_data_overlap_metrics.py"]
