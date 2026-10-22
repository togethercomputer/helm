FROM python:3.12-slim

RUN mkdir -p /script/src

WORKDIR /script/src

COPY . .

# requirements-freeze.txt is a fully resolved lock; --no-deps lets it override
# apache-beam's pillow<13 cap with the patched pillow 13.x.
RUN pip install --no-cache-dir --no-deps -r ./requirements-freeze.txt

ENTRYPOINT ["python", "compute_data_overlap_metrics.py"]
