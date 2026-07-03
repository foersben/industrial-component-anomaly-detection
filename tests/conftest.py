import asyncio
from collections.abc import Generator

import pytest


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Provide a dedicated, session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
