# Analysis-only image: same base and packages the labels were made with,
# plus coverage tooling. Never shipped; used to build maps/ offline.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock coverage==7.10.7 pytest-cov==7.0.0
