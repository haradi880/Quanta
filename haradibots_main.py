"""Console entry point used by source installs and the Enterprise Fat Binary."""

import sys

from cli.main import main
from core.frozen_converter import run_bundled_converter
from core.runtime import configure_native_runtime


if __name__ == "__main__":
    configure_native_runtime()
    if len(sys.argv) > 1 and sys.argv[1] == "_convert-hf-to-gguf":
        raise SystemExit(run_bundled_converter(sys.argv[2:]))
    raise SystemExit(main())
