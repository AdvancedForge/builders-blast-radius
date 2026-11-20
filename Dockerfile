# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image every commit was labeled in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# The selector reads the pinned sqlglot source to place each change.
ARG SQLGLOT_COMMIT=eb11aec4be8cb2122afe713a4816662c3f0fb2fa
COPY tools/fetch_sqlglot.py /tmp/fetch_sqlglot.py
RUN python /tmp/fetch_sqlglot.py ${SQLGLOT_COMMIT} && chmod -R a+rX /opt/sqlglot

COPY maps/map.json.gz /app/maps/map.json.gz
COPY selector.py /app/selector.py
RUN chmod -R a+rX /app
ENV BR_SRC=/opt/sqlglot PYTHONDONTWRITEBYTECODE=1
CMD ["python", "/app/selector.py"]
