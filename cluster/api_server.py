"""Local FastAPI gateway for the HaradiBots Enterprise Fat Binary."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from core.auth_middleware import AuthError, validate_api_key, validate_jwt
from core.orchestrator import process_job
from core.schemas import JobEnvelope


class AuthenticationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        authorization = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key")
        try:
            if api_key:
                identity = validate_api_key(api_key)
                credential = ("api_key", api_key)
            elif authorization.startswith("Bearer "):
                token = authorization.removeprefix("Bearer ").strip()
                identity = validate_jwt(token)
                credential = ("jwt_token", token)
            else:
                return JSONResponse(
                    {"error": "authentication_failed", "message": "missing credential"},
                    status_code=401,
                )
        except AuthError as exc:
            return JSONResponse(
                {"error": "authentication_failed", "message": str(exc)},
                status_code=401,
            )
        request.state.identity = identity
        request.state.credential = credential
        return await call_next(request)


app = FastAPI(title="HaradiBots Local API", version="3.0")
app.add_middleware(AuthenticationMiddleware)
_JOB_STATUS: dict[str, dict[str, Any]] = {}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs")
async def submit_job(request: Request):
    try:
        envelope = JobEnvelope.model_validate_json(await request.body())
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        return JSONResponse(
            {"error": "invalid_envelope", "message": str(exc)},
            status_code=422,
        )

    credential_name, credential_value = request.state.credential
    supplied_value = getattr(envelope.auth, credential_name)
    if supplied_value != credential_value:
        return JSONResponse(
            {
                "error": "authentication_failed",
                "message": "header and envelope credentials must match",
            },
            status_code=401,
        )

    async def event_stream():
        async for event in process_job(envelope):
            serialized = event.model_dump(mode="json")
            _JOB_STATUS[str(envelope.job_id)] = serialized
            yield f"data: {json.dumps(serialized, separators=(',', ':'))}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/jobs/{job_id}/status")
async def job_status(job_id: str):
    status = _JOB_STATUS.get(job_id)
    if status is None:
        return JSONResponse(
            {"error": "not_found", "message": f"job {job_id} was not found"},
            status_code=404,
        )
    return status
