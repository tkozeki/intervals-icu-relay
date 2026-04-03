from __future__ import annotations

import math
import os
from datetime import date, datetime, timedelta
from typing import Any, Optional

import requests
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import Response
from requests.auth import HTTPBasicAuth

app = FastAPI(
    title="Intervals.icu Relay API",
    version="0.3.0",
)


# =========================
# Environment
# =========================

INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")
INTERVALS_API_KEY = os.getenv("INTERVALS_API_KEY", "")
INTERVALS_BASE_URL = os.getenv("INTERVALS_BASE_URL", "https://intervals.icu").rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))


# =========================
# Auth / HTTP helpers
# =========================

def verify_bearer(authorization: Optional[str]) -> None:
    if not INTERNAL_TOKEN:
        raise HTTPException(status_code=500, detail="INTERNAL_TOKEN is not configured")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    if token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def intervals_get(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    accept: str | None = None,
) -> requests.Response:
    if not INTERVALS_API_KEY:
        raise HTTPException(status_code=500, detail="INTERVALS_API_KEY is not configured")

    url = f"{INTERVALS_BASE_URL}{path}"
    headers: dict[str, str] = {}
    if accept:
        headers["Accept"] = accept

    try:
        r = requests.get(
            url,
            params=params,
            headers=headers,
            auth=HTTPBasicAuth("API_KEY", INTERVALS_API_KEY),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        return r
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 502
        detail = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=status, detail=detail)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")


def parse_date_yyyy_mm_dd(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date: {value}. Expected YYYY-MM-DD")


# =========================
# Utility
# =========================

def _to_float_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []

    out: list[float] = []
    for v in values:
        try:
            if v is None:
                out.append(float("nan"))
            else:
                out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _safe_mean(xs: list[float]) -> float | None:
    vals = [x for x in xs if not math.isnan(x)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _pct_change(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or old == 0:
        return None
    return ((new / old) - 1.0) * 100.0


def _extract_stream_map(streams_payload: Any) -> dict[str, list[float]]:
    """
    想定差異を吸収:
    1) {"watts":[...], "heartrate":[...]}
    2) [{"type":"watts","data":[...]}, ...]
    """
    if isinstance(streams_payload, dict):
        out: dict[str, list[float]] = {}
        for k, v in streams_payload.items():
            if isinstance(v, list):
                out[k] = _to_float_list(v)
        return out

    if isinstance(streams_payload, list):
        out: dict[str, list[float]] = {}
        for item in streams_payload:
            if isinstance(item, dict):
                t = item.get("type")
                data = item.get("data")
                if isinstance(t, str) and isinstance(data, list):
                    out[t] = _to_float_list(data)
        return out

    return {}


def _build_mask(
    watts: list[float],
    hr: list[float],
    velocity_smooth: list[float] | None = None,
    cadence: list[float] | None = None,
) -> list[bool]:
    n = max(len(watts), len(hr), len(velocity_smooth or []), len(cadence or []))
    mask: list[bool] = []

    for i in range(n):
        p = watts[i] if i < len(watts) else float("nan")
        h = hr[i] if i < len(hr) else float("nan")
        v = velocity_smooth[i] if velocity_smooth and i < len(velocity_smooth) else float("nan")
        c = cadence[i] if cadence and i < len(cadence) else float("nan")

        valid = (
            (not math.isnan(p) and p > 0)
            and (not math.isnan(h) and h > 0)
            and (
                (not math.isnan(v) and v > 0.5)
                or (not math.isnan(c) and c > 0)
                or p > 30
            )
        )
        mask.append(valid)

    return mask


def _masked(values: list[float], mask: list[bool]) -> list[float]:
    n = min(len(values), len(mask))
    return [values[i] for i in range(n) if mask[i] and not math.isnan(values[i])]


def _half_split(values: list[float]) -> tuple[list[float], list[float]]:
    if not values:
        return [], []
    mid = len(values) // 2
    return values[:mid], values[mid:]


def _coalesce_number(obj: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in obj and obj[key] is not None:
            try:
                return float(obj[key])
            except (TypeError, ValueError):
                continue
    return None


def _coalesce_int(obj: dict[str, Any], keys: list[str]) -> int | None:
    for key in keys:
        if key in obj and obj[key] is not None:
            try:
                return int(float(obj[key]))
            except (TypeError, ValueError):
                continue
    return None


def _date_from_activity(activity: dict[str, Any]) -> str:
    for key in ["start_date_local", "start_date", "icu_training_load_date", "activity_date"]:
        value = activity.get(key)
        if isinstance(value, str) and len(value) >= 10:
            return value[:10]
    return ""


def _start_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _infer_workout_label(activity: dict[str, Any]) -> str:
    """
    summaryベースの暫定種別推定。
    将来的には detail / intervals / blocks で置き換える前提。
    """
    name = str(activity.get("name", "")).lower()
    moving_time = _coalesce_number(activity, ["moving_time", "movingTime", "elapsed_time"])
    duration = moving_time or 0.0
    avg_power = _coalesce_number(activity, ["avg_power", "average_power", "power"])
    avg_hr = _coalesce_number(activity, ["avg_hr", "average_hr", "heartrate"])
    load = _coalesce_number(activity, ["icu_training_load", "training_load", "load"])

    if duration >= 5400 and avg_power and avg_hr:
        return "z2_long"

    if "vo2" in name:
        return "vo2max"
    if "threshold" in name:
        return "threshold"
    if "sst" in name:
        return "sst"
    if "tempo" in name:
        return "tempo"
    if "lt1" in name:
        return "lt1"
    if "recovery" in name:
        return "recovery"
    if "endurance" in name or "z2" in name:
        return "z2"

    if load is not None:
        if load < 25:
            return "recovery"
        if load < 50:
            return "z2"
        if load < 75:
            return "tempo_or_lt1"
        if load < 110:
            return "sst_or_threshold"
        return "vo2max"

    return "endurance_unknown"


def _simple_ef_from_summary(activity: dict[str, Any]) -> float | None:
    avg_power = _coalesce_number(activity, ["avg_power", "average_power", "power"])
    avg_hr = _coalesce_number(activity, ["avg_hr", "average_hr", "heartrate"])
    if avg_power is None or avg_hr in (None, 0):
        return None
    return avg_power / avg_hr


def _summarize_quality(activity: dict[str, Any]) -> dict[str, Any]:
    """
    既存 workout-quality の簡易版を維持するための summary ベース評価。
    """
    moving_time = _coalesce_number(activity, ["moving_time", "movingTime", "elapsed_time"]) or 0.0
    load = _coalesce_number(activity, ["icu_training_load", "training_load", "load"]) or 0.0
    avg_power = _coalesce_number(activity, ["avg_power", "average_power", "power"])
    avg_hr = _coalesce_number(activity, ["avg_hr", "average_hr", "heartrate"])
    label = _infer_workout_label(activity)
    simple_ef = _simple_ef_from_summary(activity)

    execution_score = None
    stability_score = None
    cardio_response_score = None
    quality_score = None

    if moving_time > 0:
        base = 50.0

        if moving_time >= 1800:
            base += 10
        if moving_time >= 3600:
            base += 10
        if load >= 40:
            base += 10
        if load >= 70:
            base += 5

        if avg_power and avg_hr:
            execution_score = min(100.0, base)
            stability_score = 70.0
            cardio_response_score = 70.0
            quality_score = min(100.0, (execution_score + stability_score + cardio_response_score) / 3.0)
        else:
            execution_score = min(100.0, base - 10)
            stability_score = None
            cardio_response_score = None
            quality_score = execution_score

    return {
        "label": label,
        "execution_score": execution_score,
        "stability_score": stability_score,
        "cardio_response_score": cardio_response_score,
        "quality_score": quality_score,
        "simple_ef": simple_ef,
    }


def _fetch_activities(start: str, end: str) -> list[dict[str, Any]]:
    """
    既存構成との互換のため、複数候補パスを試す。
    実際に使っているパスが固定なら1本に絞ってOK。
    """
    candidate_paths = [
        "/api/v1/activities",
        "/api/v1/activities.json",
        "/api/v1/athlete/0/activities",
        "/api/v1/athlete/0/activities.json",
    ]

    last_error: HTTPException | None = None

    for path in candidate_paths:
        try:
            r = intervals_get(path, params={"start": start, "end": end})
            data = r.json()
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
            if isinstance(data, dict):
                for key in ["activities", "data", "items"]:
                    v = data.get(key)
                    if isinstance(v, list):
                        return [x for x in v if isinstance(x, dict)]
        except HTTPException as e:
            last_error = e
            continue

    if last_error:
        raise last_error
    raise HTTPException(status_code=502, detail="Failed to fetch activities")


def _fetch_wellness(start: str, end: str) -> list[dict[str, Any]]:
    candidate_paths = [
        "/api/v1/wellness",
        "/api/v1/wellness.json",
        "/api/v1/athlete/0/wellness",
        "/api/v1/athlete/0/wellness.json",
    ]

    last_error: HTTPException | None = None

    for path in candidate_paths:
        try:
            r = intervals_get(path, params={"start": start, "end": end})
            data = r.json()
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
            if isinstance(data, dict):
                for key in ["wellness", "data", "items"]:
                    v = data.get(key)
                    if isinstance(v, list):
                        return [x for x in v if isinstance(x, dict)]
        except HTTPException as e:
            last_error = e
            continue

    if last_error and last_error.status_code != 404:
        raise last_error

    return []


# =========================
# Health
# =========================

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# =========================
# Raw intervals passthrough
# =========================

@app.get("/intervals/activities")
def get_activities(
    start: str,
    end: str,
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)
    return _fetch_activities(start, end)


@app.get("/intervals/wellness")
def get_wellness(
    start: str,
    end: str,
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)
    return _fetch_wellness(start, end)


@app.get("/intervals/activity/{activity_id}/streams")
def get_activity_streams(
    activity_id: str,
    types: str | None = Query(
        default=None,
        description="Comma-separated stream types, e.g. watts,heartrate,cadence,time,distance",
    ),
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    params: dict[str, Any] = {}
    if types:
        params["types"] = types

    r = intervals_get(f"/api/v1/activity/{activity_id}/streams.json", params=params)
    return r.json()


@app.get("/intervals/activity/{activity_id}/streams.csv")
def get_activity_streams_csv(
    activity_id: str,
    types: str | None = Query(default=None),
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    params: dict[str, Any] = {}
    if types:
        params["types"] = types

    r = intervals_get(
        f"/api/v1/activity/{activity_id}/streams.csv",
        params=params,
        accept="text/csv",
    )
    return Response(
        content=r.content,
        media_type=r.headers.get("content-type", "text/csv"),
        headers={
            "Content-Disposition": f'attachment; filename="{activity_id}_streams.csv"',
        },
    )


@app.get("/intervals/activity/{activity_id}/file")
def get_activity_original_file(
    activity_id: str,
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    r = intervals_get(f"/api/v1/activity/{activity_id}/file")
    return Response(
        content=r.content,
        media_type=r.headers.get("content-type", "application/octet-stream"),
        headers={
            "Content-Disposition": f'attachment; filename="{activity_id}.bin"',
        },
    )


@app.get("/intervals/activity/{activity_id}/fit-file")
def get_activity_fit_file(
    activity_id: str,
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    r = intervals_get(f"/api/v1/activity/{activity_id}/fit-file")
    return Response(
        content=r.content,
        media_type=r.headers.get("content-type", "application/octet-stream"),
        headers={
            "Content-Disposition": f'attachment; filename="{activity_id}.fit.gz"',
        },
    )


# =========================
# Existing analysis endpoints
# =========================

@app.get("/analysis/training-quality")
def analyze_training_quality(
    start: str,
    end: str,
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    activities = _fetch_activities(start, end)
    wellness = _fetch_wellness(start, end)

    moving_activities = []
    high_load_days: set[str] = set()
    total_duration_seconds = 0.0
    workout_scores: list[float] = []
    ef_values: list[float] = []

    for a in activities:
        moving_time = _coalesce_number(a, ["moving_time", "movingTime", "elapsed_time"])
        if moving_time is None or moving_time <= 0:
            continue

        moving_activities.append(a)
        total_duration_seconds += moving_time

        load = _coalesce_number(a, ["icu_training_load", "training_load", "load"])
        day = _date_from_activity(a)
        if load is not None and load >= 60 and day:
            high_load_days.add(day)

        q = _summarize_quality(a)
        if q["quality_score"] is not None:
            workout_scores.append(float(q["quality_score"]))
        if q["simple_ef"] is not None:
            ef_values.append(float(q["simple_ef"]))

    sessions = len(moving_activities)
    avg_workout_quality = _safe_mean(workout_scores)
    avg_endurance_ef = _safe_mean(ef_values)

    quality_score = None
    if sessions > 0:
        base = 50.0
        if avg_workout_quality is not None:
            base = 0.7 * avg_workout_quality + 0.3 * base
        if len(high_load_days) >= 2:
            base += 5
        if len(high_load_days) >= 4:
            base += 5
        quality_score = min(100.0, base)

    return {
        "start": start,
        "end": end,
        "sessions": sessions,
        "total_duration_seconds": total_duration_seconds,
        "high_load_days": len(high_load_days),
        "avg_workout_quality": avg_workout_quality,
        "avg_endurance_ef": avg_endurance_ef,
        "wellness_entries": len(wellness),
        "quality_score": quality_score,
        "notes": [
            "High load day threshold is 60",
            "Only activities with moving_time are counted",
            "This is summary-based analysis",
        ],
    }


@app.get("/analysis/workout-quality")
def analyze_workout_quality(
    start: str,
    end: str,
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    activities = _fetch_activities(start, end)
    results: list[dict[str, Any]] = []

    for a in activities:
        moving_time = _coalesce_number(a, ["moving_time", "movingTime", "elapsed_time"])
        if moving_time is None or moving_time <= 0:
            continue

        q = _summarize_quality(a)
        results.append({
            "activity_id": a.get("id") or a.get("activity_id"),
            "date": _date_from_activity(a),
            "name": a.get("name"),
            "moving_time": moving_time,
            **q,
            "analysis_basis": "summary",
        })

    return {
        "start": start,
        "end": end,
        "count": len(results),
        "activities": results,
        "notes": [
            "Current workout-quality is provisional and summary-based",
            "Block-level evaluation for repeated LT1/tempo/SST is not implemented yet",
        ],
    }


@app.get("/analysis/endurance-efficiency")
def analyze_endurance_efficiency(
    start: str,
    end: str,
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    activities = _fetch_activities(start, end)
    results: list[dict[str, Any]] = []

    for a in activities:
        moving_time = _coalesce_number(a, ["moving_time", "movingTime", "elapsed_time"]) or 0.0
        label = _infer_workout_label(a)
        simple_ef = _simple_ef_from_summary(a)

        if moving_time <= 0:
            continue

        is_endurance_candidate = label in {
            "z2",
            "z2_long",
            "lt1",
            "tempo",
            "tempo_or_lt1",
            "endurance_unknown",
        }
        if not is_endurance_candidate:
            continue

        results.append({
            "activity_id": a.get("id") or a.get("activity_id"),
            "date": _date_from_activity(a),
            "name": a.get("name"),
            "label": label,
            "duration_seconds": moving_time,
            "simple_ef": simple_ef,
            "decoupling_pct": None,
            "hr_drift_pct": None,
            "analysis_basis": "summary",
        })

    ef_values = [r["simple_ef"] for r in results if r["simple_ef"] is not None]

    return {
        "start": start,
        "end": end,
        "count": len(results),
        "avg_simple_ef": _safe_mean([float(x) for x in ef_values]) if ef_values else None,
        "activities": results,
        "notes": [
            "Current endurance-efficiency is provisional and summary-based",
            "Decoupling and HR drift require stream-level detail and are not calculated here",
        ],
    }


@app.get("/analysis/training-quality-timeseries")
def analyze_training_quality_timeseries(
    start: str,
    end: str,
    granularity: str = Query(default="week", pattern="^(day|week)$"),
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    activities = _fetch_activities(start, end)
    wellness = _fetch_wellness(start, end)

    start_d = parse_date_yyyy_mm_dd(start)
    end_d = parse_date_yyyy_mm_dd(end)

    buckets: dict[str, dict[str, Any]] = {}

    def bucket_key_for(d: date) -> str:
        if granularity == "day":
            return d.isoformat()
        return _start_of_week(d).isoformat()

    cursor = start_d
    while cursor <= end_d:
        key = bucket_key_for(cursor)
        if key not in buckets:
            buckets[key] = {
                "period_start": key,
                "sessions": 0,
                "total_duration_seconds": 0.0,
                "high_load_days_set": set(),
                "workout_scores": [],
                "endurance_efs": [],
                "wellness_entries": 0,
            }
        cursor += timedelta(days=1)

    for a in activities:
        moving_time = _coalesce_number(a, ["moving_time", "movingTime", "elapsed_time"])
        if moving_time is None or moving_time <= 0:
            continue

        day_str = _date_from_activity(a)
        if not day_str:
            continue

        d = parse_date_yyyy_mm_dd(day_str)
        key = bucket_key_for(d)
        if key not in buckets:
            continue

        b = buckets[key]
        b["sessions"] += 1
        b["total_duration_seconds"] += moving_time

        load = _coalesce_number(a, ["icu_training_load", "training_load", "load"])
        if load is not None and load >= 60:
            b["high_load_days_set"].add(day_str)

        q = _summarize_quality(a)
        if q["quality_score"] is not None:
            b["workout_scores"].append(float(q["quality_score"]))
        if q["simple_ef"] is not None:
            label = q["label"]
            if label in {"z2", "z2_long", "lt1", "tempo", "tempo_or_lt1", "endurance_unknown"}:
                b["endurance_efs"].append(float(q["simple_ef"]))

    for w in wellness:
        day_str = ""
        for key in ["date", "day", "wellness_date"]:
            val = w.get(key)
            if isinstance(val, str) and len(val) >= 10:
                day_str = val[:10]
                break
        if not day_str:
            continue

        d = parse_date_yyyy_mm_dd(day_str)
        key = bucket_key_for(d)
        if key in buckets:
            buckets[key]["wellness_entries"] += 1

    out: list[dict[str, Any]] = []
    for key in sorted(buckets.keys()):
        b = buckets[key]
        avg_workout_quality = _safe_mean(b["workout_scores"])
        avg_endurance_ef = _safe_mean(b["endurance_efs"])

        quality_score = None
        if b["sessions"] > 0:
            base = 50.0
            if avg_workout_quality is not None:
                base = 0.7 * avg_workout_quality + 0.3 * base
            if len(b["high_load_days_set"]) >= 2:
                base += 5
            if len(b["high_load_days_set"]) >= 4:
                base += 5
            quality_score = min(100.0, base)

        out.append({
            "period_start": b["period_start"],
            "sessions": b["sessions"],
            "total_duration_seconds": b["total_duration_seconds"],
            "high_load_days": len(b["high_load_days_set"]),
            "avg_workout_quality": avg_workout_quality,
            "avg_endurance_ef": avg_endurance_ef,
            "wellness_entries": b["wellness_entries"],
            "quality_score": quality_score,
        })

    return {
        "start": start,
        "end": end,
        "granularity": granularity,
        "periods": out,
        "notes": [
            "High load day threshold is 60",
            "Only activities with moving_time are counted",
            "Time-series quality is summary-based",
        ],
    }


# =========================
# New detailed analysis endpoint
# =========================

@app.get("/analysis/workout-detail/{activity_id}")
def analyze_workout_detail(
    activity_id: str,
    warmup_exclude_seconds: int = Query(default=600, ge=0, le=3600),
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    r = intervals_get(
        f"/api/v1/activity/{activity_id}/streams.json",
        params={"types": "time,watts,heartrate,cadence,velocity_smooth"},
    )
    payload = r.json()
    smap = _extract_stream_map(payload)

    watts = smap.get("watts", [])
    heartrate = smap.get("heartrate", [])
    cadence = smap.get("cadence", [])
    velocity_smooth = smap.get("velocity_smooth", [])
    time_stream = smap.get("time", [])

    available_streams = sorted([k for k, v in smap.items() if v])

    if not watts or not heartrate:
        return {
            "activity_id": activity_id,
            "analysis_basis": "unavailable",
            "available_streams": available_streams,
            "limitations": [
                "Required streams are missing",
                "At least watts and heartrate are needed for detailed EF/drift analysis",
                "This can happen for activities whose full detail is not available via API",
            ],
        }

    n = max(len(watts), len(heartrate), len(cadence), len(velocity_smooth), len(time_stream))

    warmup_mask: list[bool] = []
    for i in range(n):
        t = time_stream[i] if i < len(time_stream) and not math.isnan(time_stream[i]) else i
        warmup_mask.append(t >= warmup_exclude_seconds)

    moving_mask = _build_mask(
        watts=watts,
        hr=heartrate,
        velocity_smooth=velocity_smooth,
        cadence=cadence,
    )

    final_mask = [
        (warmup_mask[i] if i < len(warmup_mask) else True)
        and (moving_mask[i] if i < len(moving_mask) else False)
        for i in range(n)
    ]

    p = _masked(watts, final_mask)
    h = _masked(heartrate, final_mask)

    if len(p) < 300 or len(h) < 300:
        return {
            "activity_id": activity_id,
            "analysis_basis": "unavailable",
            "available_streams": available_streams,
            "limitations": [
                "Not enough valid paired watts/heartrate samples after filtering",
                "Need sustained moving data for meaningful EF/drift analysis",
            ],
        }

    avg_power = _safe_mean(p)
    avg_hr = _safe_mean(h)
    ef = (avg_power / avg_hr) if (avg_power is not None and avg_hr not in (None, 0)) else None

    p1, p2 = _half_split(p)
    h1, h2 = _half_split(h)

    p1m = _safe_mean(p1)
    p2m = _safe_mean(p2)
    h1m = _safe_mean(h1)
    h2m = _safe_mean(h2)

    ef1 = (p1m / h1m) if (p1m is not None and h1m not in (None, 0)) else None
    ef2 = (p2m / h2m) if (p2m is not None and h2m not in (None, 0)) else None

    decoupling_pct = _pct_change(ef1, ef2)
    hr_drift_pct = _pct_change(h1m, h2m)

    interpretation: list[str] = []
    if decoupling_pct is not None:
        if decoupling_pct >= 5:
            interpretation.append("後半でpower/HRが改善。前半立ち上がりの影響が大きい可能性")
        elif decoupling_pct > -3:
            interpretation.append("Pa:Hrは安定。持久系として良好")
        elif decoupling_pct > -5:
            interpretation.append("軽度のデカップリング。許容範囲")
        else:
            interpretation.append("デカップリングがやや大きい。耐久性、疲労、補給、暑熱を要確認")

    if hr_drift_pct is not None:
        if hr_drift_pct < 3:
            interpretation.append("心拍ドリフトは小さい")
        elif hr_drift_pct < 5:
            interpretation.append("心拍ドリフトは中程度")
        else:
            interpretation.append("心拍ドリフトは大きめ")

    return {
        "activity_id": activity_id,
        "analysis_basis": "streams",
        "available_streams": available_streams,
        "samples_used": len(p),
        "warmup_exclude_seconds": warmup_exclude_seconds,
        "whole_activity": {
            "avg_power": avg_power,
            "avg_hr": avg_hr,
            "ef": ef,
            "first_half_ef": ef1,
            "second_half_ef": ef2,
            "decoupling_pct": decoupling_pct,
            "hr_drift_pct": hr_drift_pct,
        },
        "interpretation": interpretation,
        "limitations": [
            "This endpoint evaluates the whole filtered activity",
            "Repeated LT1/tempo/SST block-by-block evaluation is not implemented yet",
            "No explicit lap/interval segmentation yet",
        ],
    }


# =========================
# Optional future stub
# =========================

@app.get("/analysis/endurance-efficiency-detail")
def analyze_endurance_efficiency_detail(
    start: str,
    end: str,
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    activities = _fetch_activities(start, end)
    results: list[dict[str, Any]] = []

    for a in activities:
        activity_id = a.get("id") or a.get("activity_id")
        moving_time = _coalesce_number(a, ["moving_time", "movingTime", "elapsed_time"]) or 0.0
        label = _infer_workout_label(a)

        if not activity_id or moving_time <= 0:
            continue

        is_candidate = label in {"z2", "z2_long", "lt1", "tempo", "tempo_or_lt1", "endurance_unknown"}
        if not is_candidate:
            continue

        # まずはsummary fallbackで一覧化し、
        # 必要なら個別に /analysis/workout-detail/{activity_id} を呼ぶ運用にする
        results.append({
            "activity_id": activity_id,
            "date": _date_from_activity(a),
            "name": a.get("name"),
            "label": label,
            "duration_seconds": moving_time,
            "analysis_basis": "summary",
            "simple_ef": _simple_ef_from_summary(a),
            "decoupling_pct": None,
            "hr_drift_pct": None,
            "note": "Use /analysis/workout-detail/{activity_id} for stream-based detail",
        })

    return {
        "start": start,
        "end": end,
        "count": len(results),
        "activities": results,
        "notes": [
            "This endpoint is currently a summary-based candidate list",
            "Detailed EF/decoupling/HR drift should be checked per activity via workout-detail",
        ],
    }
    
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
