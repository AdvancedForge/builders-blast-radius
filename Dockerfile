# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image every commit was labeled in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# The selector reads the pinned sqlglot source to place each change.
ARG SQLGLOT_COMMIT=eb11aec4be8cb2122afe713a4816662c3f0fb2fa
COPY tools/fetch_sqlglot.py /tmp/fetch_sqlglot.py
RUN python /tmp/fetch_sqlglot.py ${SQLGLOT_COMMIT} \
 && python -m compileall -q /opt/sqlglot/sqlglot /opt/sqlglot/tests; chmod -R a+rX /opt/sqlglot

# Test dependencies as labeled, for small live runs.
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

COPY maps/ /app/maps/
COPY selector.py live_plugin.py tpc_fp.py import_fp.py fastlive.py tools/revdeps.py tools/families.py /app/
# Fingerprint of what the TPC-DS executor test computes at the pinned commit.
RUN cd /opt/sqlglot && PYTHONHASHSEED=0 SKIP_INTEGRATION=0 PYTHONPATH=/opt/sqlglot \
    python /app/tpc_fp.py base /app/tpcds_fp.json && test -s /app/tpcds_fp.json \
 && PYTHONHASHSEED=0 PYTHONPATH=/opt/sqlglot python /app/import_fp.py base /app/import_fp.json \
 && python /app/revdeps.py /opt/sqlglot /app/revdeps.json \
 && python /app/families.py /tmp/families.json && mv /tmp/families.json /app/families.json
RUN chmod -R a+rX /app
ENV BR_SRC=/opt/sqlglot PYTHONDONTWRITEBYTECODE=1
CMD ["python", "/app/selector.py"]
