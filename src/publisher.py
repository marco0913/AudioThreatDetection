"""HTTP publisher -- posts AlertEvent JSON to the subscriber service.

Transport is isolated here so swapping to Pub/Sub, SNS, or any other
broker requires changing only this module.
"""

import os

import httpx

from src.models import AlertEvent

SUBSCRIBER_URL = os.getenv("SUBSCRIBER_URL", "http://localhost:8765/alerts")


async def publish(event: AlertEvent, url: str = SUBSCRIBER_URL) -> bool:
    """Publish an AlertEvent to the subscriber. Returns True on success."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                url,
                content=event.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                print(f"[PUBLISH] POST {url} -> 200 OK")
                return True
            else:
                print(f"[PUBLISH] POST {url} -> {resp.status_code} {resp.text}")
                return False
    except httpx.ConnectError:
        print(
            f"[PUBLISH] WARNING: subscriber not reachable at {url} "
            "-- alert will not be delivered"
        )
        return False
