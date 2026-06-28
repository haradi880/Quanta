import asyncio
import socket
from pathlib import Path

import pytest

from cli.doctor import _verify_resp_operations
from telemetry.redis_manager import LocalRedisManager


@pytest.mark.integration
def test_bundled_garnet_real_resp_contract():
    binary = (
        Path(__file__).resolve().parents[1]
        / "build"
        / "vendor"
        / "garnet"
        / "GarnetServer.exe"
    )
    if not binary.is_file():
        pytest.skip("deterministic Garnet vendor payload is not populated")

    async def scenario():
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        manager = LocalRedisManager(binary_path=str(binary), port=port)
        url = await manager.start(timeout_seconds=15)
        try:
            result = await _verify_resp_operations(url)
        finally:
            await manager.stop()
        assert result["PING"] == "PONG"
        assert result["HSET"] == 2
        assert result["HGETALL"] == {"gpu": "0", "status": "healthy"}
        assert len(result["SCAN"]) == 1
        assert manager.process is None

    asyncio.run(scenario())
