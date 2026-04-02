import os
from typing import Any

import requests
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse


INTERNAL_TOKEN = os.environ["INTERNAL_TOKEN"]
INTERVALS_API_KEY = os.environ["INTERVALS_API_KEY"]
INTERVALS_BASE_URL = os.environ.get("INTERVALS_BASE_URL", "https://intervals.icu/api/v1")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "20"))

app = FastAPI(
    title="Intervals.icu Relay API",
    version="1.0.0",
    description=(
        "Relay API for ChatGPT Actions. "
        "ChatGPT authenticates with a Bearer token, and this service authenticates "
        "to Intervals.icu with Basic auth using your Intervals API key."
    ),
)


def require_bearer(authorization: str | None) -> None:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")



def intervals_get(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{INTERVALS_BASE_URL}{path}"
    try:
        response = requests.get(
            url,
            params=params,
            auth=("API_KEY", INTERVALS_API_KEY),
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Intervals request failed: {exc}") from exc

    if response.status_code >= 400:
        detail: Any
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Intervals API returned an error",
                "status_code": response.status_code,
                "body": detail,
            },
        )

    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Intervals API did not return JSON") from exc


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Intervals.icu relay API is running.",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/healthz",
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/intervals/activities")
def get_activities(
    start: str = Query(..., description="Start date in YYYY-MM-DD"),
    end: str = Query(..., description="End date in YYYY-MM-DD"),
    authorization: str | None = Header(None),
):
    require_bearer(authorization)
    data = intervals_get("/athlete/0/activities", {"oldest": start, "newest": end})
    return JSONResponse(content=data)


@app.get("/intervals/wellness")
def get_wellness(
    start: str = Query(..., description="Start date in YYYY-MM-DD"),
    end: str = Query(..., description="End date in YYYY-MM-DD"),
    authorization: str | None = Header(None),
):
    require_bearer(authorization)
    data = intervals_get("/athlete/0/wellness", {"oldest": start, "newest": end})
    return JSONResponse(content=data)


@app.get("/analysis/training-quality")
def analyze_training_quality(
    start: str = Query(..., description="Start date in YYYY-MM-DD"),
    end: str = Query(..., description="End date in YYYY-MM-DD"),
    authorization: str | None = Header(None),
):
    require_bearer(authorization)

    activities = intervals_get("/athlete/0/activities", {"oldest": start, "newest": end})
    wellness = intervals_get("/athlete/0/wellness", {"oldest": start, "newest": end})

    valid_activities = [a for a in activities if a.get("moving_time")]

    sessions = len(valid_activities)
    total_duration = sum((a.get("moving_time") or 0) for a in valid_activities)
    high_load_days = sum(
        1 for a in valid_activities if (a.get("icu_training_load") or 0) >= 60
    )

    strengths: list[str] = []
    concerns: list[str] = []

    if sessions >= 4:
        strengths.append("実施頻度は十分です")
    else:
        concerns.append("実施頻度が少なめです")

    if high_load_days <= 2:
        strengths.append("高負荷日の密度は過剰ではありません")
    else:
        concerns.append("高負荷日が多く、回復不足の可能性があります")

    quality_score = max(
        0,
        min(100, 70 + min(sessions, 5) * 4 - max(0, high_load_days - 2) * 8),
    )

    return {
        "summary": (
            f"{start}〜{end}に {sessions} セッション、"
            f"総運動時間は約 {round(total_duration / 3600, 1)} 時間です。"
        ),
        "quality_score": quality_score,
        "strengths": strengths,
        "concerns": concerns,
        "raw_counts": {
            "sessions": sessions,
            "high_load_days": high_load_days,
            "total_duration_seconds": total_duration,
            "wellness_entries": len(wellness),
        },
    }
