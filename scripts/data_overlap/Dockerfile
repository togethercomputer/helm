FROM python:3.12-slim

RUN mkdir -p /script/src

WORKDIR /script/src

COPY . .

# requirements-freeze.txt is a complete, fully resolved lock. --no-deps is needed
# because apache-beam 2.77.0 (latest) declares pillow<13, while every Pillow 12.x
# release has unfixed advisories (e.g. CVE-2026-96639) that are only fixed in
# 13.0.0. Beam only imports PIL in its ML/inference modules, which this script
# does not use.
RUN pip install --no-cache-dir --no-deps -r ./requirements-freeze.txt

ENTRYPOINT ["python", "compute_data_overlap_metrics.py"]
