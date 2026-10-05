# Contributing

PyHREBSD is currently an early research release. Bug reports and reproducible
test cases are welcome through GitHub Issues.

Before preparing a code contribution, open an issue to discuss its scope and
licensing. The repository does not yet use a contributor license agreement,
so unsolicited pull requests may not be accepted. Do not submit code copied
from software whose license is incompatible with the PyHREBSD license.

For local development:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python -m pytest
```

Use a small synthetic or redistributable test fixture. Do not commit private
H5OINA scans, proprietary detector data, generated result directories, or
publication PDFs.
