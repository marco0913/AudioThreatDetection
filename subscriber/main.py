"""Alert subscriber service — receives and persists AlertEvent JSON over HTTP."""

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Add project root to path so we can import src.models
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import AlertEvent

app = FastAPI(title="Alert Subscriber")

ALERTS_FILE = Path(__file__).resolve().parent.parent / "alerts.jsonl"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/alerts")
async def receive_alert(event: AlertEvent):
    # Log to console
    print(
        f"[ALERT RECEIVED] {event.alert_type} | "
        f"severity={event.severity} | "
        f"score={event.fusion_score:.2f} | "
        f"incident={event.incident_id} | "
        f"source={event.source_file} | "
        f"reason={event.reasoning[:80]}"
    )

    # Persist to JSONL
    with open(ALERTS_FILE, "a", encoding="utf-8") as f:
        f.write(event.model_dump_json() + "\n")

    count = sum(1 for _ in open(ALERTS_FILE, encoding="utf-8"))
    print(f"[PERSISTED] alerts.jsonl ({count} events)")

    return JSONResponse(
        status_code=200,
        content={"status": "received", "event_id": event.event_id},
    )


if __name__ == "__main__":
    import uvicorn

    print(f"[SUBSCRIBER] Alerts will be persisted to: {ALERTS_FILE}")
    uvicorn.run(app, host="0.0.0.0", port=8765)
