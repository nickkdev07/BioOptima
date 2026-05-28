from __future__ import annotations

import glob
import py_compile

paths: list[str] = []
paths.append("app.py")
paths += glob.glob("pages/*.py")
paths += glob.glob("src/*.py")

failed: list[tuple[str, str]] = []
for p in paths:
    try:
        py_compile.compile(p, doraise=True)
    except Exception as e:  # noqa: BLE001
        failed.append((p, str(e)))

if failed:
    print("FAIL:")
    for p, err in failed:
        print(" -", p, "=>", err)
    raise SystemExit(1)

print("OK compiled", len(paths), "files")

