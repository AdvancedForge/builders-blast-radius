"""Download sqlglot at the pinned commit into /opt/sqlglot (image build step)."""
import io, shutil, sys, tarfile, urllib.request

commit = sys.argv[1]
url = f"https://github.com/tobymao/sqlglot/archive/{commit}.tar.gz"
for attempt in range(3):
    try:
        data = urllib.request.urlopen(url, timeout=120).read()
        break
    except Exception as e:  # noqa: BLE001
        print("retrying download:", e, file=sys.stderr)
else:
    raise SystemExit("could not download sqlglot")
with tarfile.open(fileobj=io.BytesIO(data)) as tar:
    tar.extractall("/opt", filter="data")
shutil.move(f"/opt/sqlglot-{commit}", "/opt/sqlglot")
