"""Console entry point used by source installs and the Enterprise Fat Binary."""

from cli.main import main
from core.runtime import configure_native_runtime


if __name__ == "__main__":
    configure_native_runtime()
    raise SystemExit(main())
