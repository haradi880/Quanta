"""Internal node inventory endpoint; the launcher enforces client-certificate TLS."""

from fastapi import FastAPI

from core.profiler import snapshot


app = FastAPI(title="HaradiBots Cluster Node", docs_url=None, redoc_url=None)


@app.get("/health/gpu")
async def gpu_health():
    profile = snapshot()
    return {
        "hardware_fault": None,
        "gpus": profile["gpus"],
    }
