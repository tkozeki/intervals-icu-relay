import os
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean
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
    version="1.1.0",
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


def parse_ymd(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {s}. Use YYYY-MM-DD") from exc


def date_to_ymd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def safe_num(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def is_valid_activity(activity: dict[str, Any]) -> bool:
    return bool(activity.get("moving_time"))


def extract_activity_date(activity: dict[str, Any]) -> str | None:
    raw_date = activity.get("start_date_local") or activity.get("start_date")
    if not raw_date:
        return None
    return str(raw_date)[:10]


def extract_activity_datetime(activity: dict[str, Any]) -> datetime | None:
    raw_date = activity.get("start_date_local") or activity.get("start_date")
    if not raw_date:
        return None

    raw_date = str(raw_date)
    try:
        return datetime.fromisoformat(raw_date.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        try:
            return parse_ymd(raw_date[:10])
        except HTTPException:
            return None


def detect_workout_type(activity: dict[str, Any]) -> str:
    name = (activity.get("name") or "").lower()
    activity_type = (activity.get("type") or "").lower()
    moving_time = safe_num(activity.get("moving_time"))
    load = safe_num(activity.get("icu_training_load"))
    avg_power = safe_num(activity.get("avg_power"))
    avg_hr = safe_num(activity.get("avg_hr"))

    if "sst" in name:
        return "sst"
    if "tempo" in name:
        return "tempo"
    if "lt1" in name:
        return "lt1"
    if "z2" in name or "zone 2" in name:
        return "z2"
    if "threshold" in name:
        return "threshold"
    if "vo2" in name or "vo₂" in name:
        return "vo2max"
    if "recovery" in name or "easy" in name:
        return "recovery"

    if moving_time >= 90 * 60 and load <= 90:
        return "z2_long"
    if 45 * 60 <= moving_time <= 120 * 60 and 50 <= load <= 90:
        return "tempo_or_lt1"
    if 45 * 60 <= moving_time <= 100 * 60 and 75 <= load <= 120:
        return "sst_or_threshold"
    if activity_type in {"ride", "virtualride"} and avg_power > 0 and avg_hr > 0:
        return "endurance_unknown"

    return "unknown"


def calc_simple_ef(activity: dict[str, Any]) -> float | None:
    avg_power = safe_num(activity.get("avg_power"))
    avg_hr = safe_num(activity.get("avg_hr"))

    if avg_power <= 0 or avg_hr <= 0:
        return None

    return round(avg_power / avg_hr, 3)


def is_endurance_candidate(activity: dict[str, Any]) -> bool:
    moving_time = safe_num(activity.get("moving_time"))
    avg_hr = safe_num(activity.get("avg_hr"))
    avg_power = safe_num(activity.get("avg_power"))
    wtype = detect_workout_type(activity)

    if moving_time < 40 * 60:
        return False
    if avg_hr <= 0 or avg_power <= 0:
        return False

    return wtype in {
        "z2", "z2_long", "lt1", "tempo", "sst", "tempo_or_lt1", "endurance_unknown"
    }


def calc_simple_workout_quality(activity: dict[str, Any]) -> dict[str, Any]:
    moving_time = safe_num(activity.get("moving_time"))
    load = safe_num(activity.get("icu_training_load"))
    avg_hr = safe_num(activity.get("avg_hr"))
    avg_power = safe_num(activity.get("avg_power"))
    wtype = detect_workout_type(activity)

    strengths: list[str] = []
    concerns: list[str] = []

    execution_score = 50
    stability_score = 50
    cardio_response_score = 50

    if moving_time >= 45 * 60:
        execution_score += 15
        strengths.append("十分な実施時間があります")
    elif moving_time >= 25 * 60:
        execution_score += 5
    else:
        concerns.append("実施時間が短めです")

    if load > 0:
        execution_score += 10

    if avg_power > 0:
        stability_score += 15
        strengths.append("出力データが取得されています")
    else:
        concerns.append("出力データが不足しています")

    if avg_hr > 0:
        cardio_response_score += 10
    else:
        concerns.append("心拍データが不足しています")

    if wtype in {"z2", "z2_long", "lt1", "tempo", "tempo_or_lt1"}:
        if moving_time >= 90 * 60:
            stability_score += 10
            strengths.append("持久系ワークアウトとして十分な時間です")
        if calc_simple_ef(activity) is not None:
            cardio_response_score += 10

    if wtype in {"sst", "threshold", "sst_or_threshold"}:
        if 40 * 60 <= moving_time <= 100 * 60:
            execution_score += 10
            strengths.append("SST/しきい値系として妥当な時間です")
        if load >= 70:
            stability_score += 10

    if wtype == "recovery":
        if load <= 40:
            strengths.append("回復目的として負荷は抑えられています")
        else:
            concerns.append("回復走としてはやや負荷が高い可能性があります")

    execution_score = max(0, min(100, int(execution_score)))
    stability_score = max(0, min(100, int(stability_score)))
    cardio_response_score = max(0, min(100, int(cardio_response_score)))

    quality_score = int(round(
        execution_score * 0.4 +
        stability_score * 0.35 +
        cardio_response_score * 0.25
    ))

    return {
        "type_detected": wtype,
        "quality_score": quality_score,
        "execution_score": execution_score,
        "stability_score": stability_score,
        "cardio_response_score": cardio_response_score,
        "strengths": strengths[:3],
        "concerns": concerns[:3],
        "simple_ef": calc_simple_ef(activity),
        "note": "summaryベースの暫定評価です。EF/デカップリング/ドリフトの厳密評価には詳細時系列データが必要です。",
    }


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
    start_dt = parse_ymd(start)
    end_dt = parse_ymd(end)
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start must be <= end")

    data = intervals_get("/athlete/0/activities", {"oldest": start, "newest": end})
    return JSONResponse(content=data)


@app.get("/intervals/wellness")
def get_wellness(
    start: str = Query(..., description="Start date in YYYY-MM-DD"),
    end: str = Query(..., description="End date in YYYY-MM-DD"),
    authorization: str | None = Header(None),
):
    require_bearer(authorization)
    start_dt = parse_ymd(start)
    end_dt = parse_ymd(end)
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start must be <= end")

    data = intervals_get("/athlete/0/wellness", {"oldest": start, "newest": end})
    return JSONResponse(content=data)


@app.get("/analysis/training-quality")
def analyze_training_quality(
    start: str = Query(..., description="Start date in YYYY-MM-DD"),
    end: str = Query(..., description="End date in YYYY-MM-DD"),
    authorization: str | None = Header(None),
):
    require_bearer(authorization)

    start_dt = parse_ymd(start)
    end_dt = parse_ymd(end)
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start must be <= end")

    activities = intervals_get("/athlete/0/activities", {"oldest": start, "newest": end})
    wellness = intervals_get("/athlete/0/wellness", {"oldest": start, "newest": end})

    valid_activities = [a for a in activities if is_valid_activity(a)]

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


@app.get("/analysis/workout-quality")
def analyze_workout_quality(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    authorization: str | None = Header(None),
):
    require_bearer(authorization)

    start_dt = parse_ymd(start)
    end_dt = parse_ymd(end)
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start must be <= end")

    activities = intervals_get("/athlete/0/activities", {"oldest": start, "newest": end})
    valid_activities = [a for a in activities if is_valid_activity(a)]

    workouts = []
    for a in valid_activities:
        quality = calc_simple_workout_quality(a)

        workouts.append({
            "id": a.get("id"),
            "date": extract_activity_date(a),
            "name": a.get("name"),
            "type": a.get("type"),
            "moving_time": a.get("moving_time"),
            "icu_training_load": a.get("icu_training_load"),
            "avg_power": a.get("avg_power"),
            "avg_hr": a.get("avg_hr"),
            **quality,
        })

    avg_quality = round(mean([w["quality_score"] for w in workouts]), 1) if workouts else None

    return {
        "summary": f"{start}〜{end}の {len(workouts)} 件のワークアウトを暫定評価しました。",
        "average_quality_score": avg_quality,
        "workouts": workouts,
    }


@app.get("/analysis/endurance-efficiency")
def analyze_endurance_efficiency(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    authorization: str | None = Header(None),
):
    require_bearer(authorization)

    start_dt = parse_ymd(start)
    end_dt = parse_ymd(end)
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start must be <= end")

    activities = intervals_get("/athlete/0/activities", {"oldest": start, "newest": end})
    valid_activities = [a for a in activities if is_valid_activity(a)]
    candidates = [a for a in valid_activities if is_endurance_candidate(a)]

    items = []
    ef_values = []

    for a in candidates:
        ef = calc_simple_ef(a)
        if ef is not None:
            ef_values.append(ef)

        duration_min = round(safe_num(a.get("moving_time")) / 60, 1)
        load = safe_num(a.get("icu_training_load"))

        rating = "insufficient_data"
        if ef is not None:
            if duration_min >= 90 and load <= 90:
                rating = "good"
            else:
                rating = "usable"

        items.append({
            "id": a.get("id"),
            "date": extract_activity_date(a),
            "name": a.get("name"),
            "type_detected": detect_workout_type(a),
            "duration_minutes": duration_min,
            "avg_power": a.get("avg_power"),
            "avg_hr": a.get("avg_hr"),
            "simple_ef": ef,
            "decoupling_pct": None,
            "hr_drift_pct": None,
            "rating": rating,
            "note": "現段階ではsummaryベースの簡易EFのみ。詳細版では時系列データからdecoupling/hr driftを計算します。",
        })

    avg_ef = round(mean(ef_values), 3) if ef_values else None

    return {
        "summary": f"{start}〜{end}で持久効率評価候補は {len(items)} 件です。",
        "average_simple_ef": avg_ef,
        "items": items,
    }


@app.get("/analysis/training-quality-timeseries")
def analyze_training_quality_timeseries(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    granularity: str = Query("week", description="day or week"),
    authorization: str | None = Header(None),
):
    require_bearer(authorization)

    start_dt = parse_ymd(start)
    end_dt = parse_ymd(end)
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start must be <= end")
    if granularity not in {"day", "week"}:
        raise HTTPException(status_code=400, detail="granularity must be 'day' or 'week'")

    activities = intervals_get("/athlete/0/activities", {"oldest": start, "newest": end})
    wellness = intervals_get("/athlete/0/wellness", {"oldest": start, "newest": end})

    valid_activities = [a for a in activities if is_valid_activity(a)]

    grouped_activities: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_wellness: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def bucket_key(dt: datetime) -> str:
        if granularity == "day":
            return date_to_ymd(dt)
        monday = dt - timedelta(days=dt.weekday())
        return date_to_ymd(monday)

    for a in valid_activities:
        dt = extract_activity_datetime(a)
        if dt is None:
            continue
        grouped_activities[bucket_key(dt)].append(a)

    for w in wellness:
        raw_date = w.get("id") or w.get("date") or w.get("updated")
        if not raw_date:
            continue
        raw_date = str(raw_date)[:10]
        try:
            dt = parse_ymd(raw_date)
        except HTTPException:
            continue
        grouped_wellness[bucket_key(dt)].append(w)

    current = start_dt
    points = []
    seen_keys: set[str] = set()

    while current <= end_dt:
        key = bucket_key(current)
        if key not in seen_keys:
            acts = grouped_activities.get(key, [])
            wells = grouped_wellness.get(key, [])

            sessions = len(acts)
            total_duration = sum(safe_num(a.get("moving_time")) for a in acts)
            high_load_days = sum(1 for a in acts if safe_num(a.get("icu_training_load")) >= 60)

            workout_scores = [calc_simple_workout_quality(a)["quality_score"] for a in acts]
            ef_candidates = [calc_simple_ef(a) for a in acts]
            ef_values = [x for x in ef_candidates if x is not None]

            avg_workout_quality = round(mean(workout_scores), 1) if workout_scores else None
            avg_endurance_ef = round(mean(ef_values), 3) if ef_values else None

            quality_score = max(
                0,
                min(100, 70 + min(sessions, 5) * 4 - max(0, high_load_days - 2) * 8)
            )

            points.append({
                "period_start": key,
                "sessions": sessions,
                "total_duration_seconds": int(total_duration),
                "high_load_days": high_load_days,
                "avg_workout_quality": avg_workout_quality,
                "avg_endurance_ef": avg_endurance_ef,
                "wellness_entries": len(wells),
                "quality_score": quality_score,
            })
            seen_keys.add(key)

        if granularity == "day":
            current += timedelta(days=1)
        else:
            current += timedelta(days=7)

    return {
        "summary": f"{start}〜{end}を {granularity} 単位で品質評価しました。",
        "granularity": granularity,
        "points": points,
    }

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
