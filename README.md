# HaradiBots

HaradiBots is an LLM quantization and profiling engine.

The repository is being built phase by phase from the HaradiBots v3.0 build roadmap.

## Development setup

HaradiBots requires Python 3.11 or newer. On Windows, create and activate a
virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

For a CPU-only installation, install Torch from the PyTorch CPU index before
installing the complete requirements:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Run the Phase 1 test with:

```powershell
python -m pytest tests/test_import_isolation.py
```

## Local authentication setup

Generate a development API key once:

```powershell
python scripts\gen_credentials.py
```

The command writes the plaintext key only to the ignored `.env` file and its
SHA-256 hash only to the ignored `config/credentials.json` store. It does not
print the key. To exercise JWT authentication, set a secret of at least 32
random bytes in `HARADIBOTS_JWT_SECRET`; do not commit it.
