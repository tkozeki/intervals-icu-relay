from __future__ import annotations

import math
import os
import statistics
from datetime import date, datetime, timedelta
from typing import Any, Optional

import requests
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, model_validator
from requests.auth import HTTPBasicAuth

app = FastAPI(
    title="Intervals.icu Relay API",
    version="0.4.0",
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


def _intervals_api_path(path: str) -> str:
    """
    INTERVALS_BASE_URL が
    - https://intervals.icu
    - https://intervals.icu/api/v1
    のどちらでも動くように正規化する。
    """
    normalized = path if path.startswith("/") else f"/{path}"

    if INTERVALS_BASE_URL.endswith("/api/v1"):
        if normalized.startswith("/api/v1/"):
            return normalized.removeprefix("/api/v1")
        return normalized

    if normalized.startswith("/api/v1/"):
        return normalized

    return f"/api/v1{normalized}"


def intervals_get(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    accept: str | None = None,
) -> requests.Response:
    if not INTERVALS_API_KEY:
        raise HTTPException(status_code=500, detail="INTERVALS_API_KEY is not configured")

    url = f"{INTERVALS_BASE_URL}{_intervals_api_path(path)}"
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
# Models
# =========================

class WorkoutBlock(BaseModel):
    start_sec: int = Field(..., ge=0, description="Block start time in seconds from activity start")
    end_sec: int = Field(..., gt=0, description="Block end time in seconds from activity start")
    label: Optional[str] = Field(default=None, description="Optional block label")

    @model_validator(mode="after")
    def validate_range(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        return self


class WorkoutBlockQualityRequest(BaseModel):
    activity_id: str = Field(..., description="Intervals.icu activity ID")
    blocks: list[WorkoutBlock] = Field(..., min_length=1, description="User-specified blocks for analysis")


class BlockAnalysisResult(BaseModel):
    index: int
    label: str
    start_sec: int
    end_sec: int
    duration_sec: int
    avg_power: Optional[float] = None
    avg_hr: Optional[float] = None
    avg_cadence: Optional[float] = None
    avg_speed_mps: Optional[float] = None
    ef: Optional[float] = None
    power_cv: Optional[float] = None
    hr_cv: Optional[float] = None
    cadence_cv: Optional[float] = None
    power_variation_pct: Optional[float] = None
    negative_split_power_pct: Optional[float] = None
    negative_split_hr_pct: Optional[float] = None
    drift_pct: Optional[float] = None
    decoupling_pct: Optional[float] = None
    stability_score: Optional[float] = None
    block_quality_label: Optional[str] = None


class BlockDropItem(BaseModel):
    label: str
    drop_pct: Optional[float] = None


class BetweenBlockComparison(BaseModel):
    power_drop_from_first_pct: list[BlockDropItem]
    ef_drop_from_first_pct: list[BlockDropItem]


class BlockSummary(BaseModel):
    best_block_label: Optional[str] = None
    worst_block_label: Optional[str] = None
    consistency_assessment: Optional[str] = None
    overall_block_quality: Optional[str] = None


class WorkoutBlockQualityResponse(BaseModel):
    activity_id: str
    analysis_basis: str
    available_streams: list[str]
    block_count: int
    blocks_analyzed: list[BlockAnalysisResult]
    between_block_comparison: Optional[BetweenBlockComparison] = None
    summary: Optional[BlockSummary] = None
    limitations: list[str]


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


def _safe_std(xs: list[float]) -> float | None:
    vals = [x for x in xs if not math.isnan(x)]
    if len(vals) < 2:
        return 0.0
    return statistics.pstdev(vals)


def _safe_cv(xs: list[float]) -> float | None:
    mean_v = _safe_mean(xs)
    if mean_v is None or mean_v == 0:
        return None
    std_v = _safe_std(xs)
    if std_v is None:
        return None
    return std_v / mean_v


def _round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)


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
    summaryベース評価。
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
    旧版で実際に動いていた取得方法に合わせる。
    /athlete/0/activities + oldest/newest
    """
    r = intervals_get("/athlete/0/activities", params={"oldest": start, "newest": end})
    data = r.json()

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ["activities", "data", "items"]:
            v = data.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]

    raise HTTPException(status_code=502, detail="Unexpected activities response")


def _fetch_wellness(start: str, end: str) -> list[dict[str, Any]]:
    """
    旧版で実際に動いていた取得方法に合わせる。
    /athlete/0/wellness + oldest/newest
    """
    r = intervals_get("/athlete/0/wellness", params={"oldest": start, "newest": end})
    data = r.json()

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ["wellness", "data", "items"]:
            v = data.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]

    return []


# =========================
# Block analysis helpers
# =========================

REQUIRED_BLOCK_STREAMS = {"time", "watts", "heartrate"}


def _slice_stream_by_time(
    time_stream: list[float],
    value_stream: list[float] | None,
    start_sec: int,
    end_sec: int,
) -> list[float]:
    if not value_stream:
        return []

    out: list[float] = []
    n = min(len(time_stream), len(value_stream))
    for i in range(n):
        t = time_stream[i]
        v = value_stream[i]
        if math.isnan(t) or math.isnan(v):
            continue
        if start_sec <= t < end_sec:
            out.append(v)
    return out


def _split_block_stream(
    time_stream: list[float],
    value_stream: list[float] | None,
    start_sec: int,
    end_sec: int,
) -> dict[str, list[float]]:
    if not value_stream:
        return {"first": [], "second": []}

    mid = start_sec + (end_sec - start_sec) / 2.0
    first_half: list[float] = []
    second_half: list[float] = []

    n = min(len(time_stream), len(value_stream))
    for i in range(n):
        t = time_stream[i]
        v = value_stream[i]
        if math.isnan(t) or math.isnan(v):
            continue
        if start_sec <= t < end_sec:
            if t < mid:
                first_half.append(v)
            else:
                second_half.append(v)

    return {"first": first_half, "second": second_half}


def _compute_stability_score(
    duration_sec: int,
    power_variation_pct: float | None,
    decoupling_pct: float | None,
    cadence_cv: float | None,
) -> float | None:
    score = 100.0

    if power_variation_pct is not None:
        score -= min(max(power_variation_pct, 0.0), 20.0) * 2.0

    if decoupling_pct is not None:
        score -= min(abs(decoupling_pct), 15.0) * 2.5

    if cadence_cv is not None:
        score -= min(cadence_cv * 100.0, 15.0) * 1.2

    if duration_sec < 300:
        score -= 10.0
    elif duration_sec < 600:
        score -= 4.0

    return max(0.0, min(100.0, score))


def _label_block_quality(stability_score: float | None, decoupling_pct: float | None) -> str | None:
    if stability_score is None:
        return None

    dec = abs(decoupling_pct) if decoupling_pct is not None else None

    if stability_score >= 85 and (dec is None or dec <= 5):
        return "good"
    if stability_score >= 70 and (dec is None or dec <= 8):
        return "acceptable"
    if stability_score >= 55:
        return "mixed"
    return "poor"


def _analyze_single_block(
    index: int,
    block: WorkoutBlock,
    smap: dict[str, list[float]],
) -> BlockAnalysisResult:
    label = block.label or f"block_{index + 1}"

    time_stream = smap.get("time", [])
    watts_stream = smap.get("watts", [])
    hr_stream = smap.get("heartrate", [])
    cadence_stream = smap.get("cadence", [])
    speed_stream = smap.get("velocity_smooth", [])

    watts = _slice_stream_by_time(time_stream, watts_stream, block.start_sec, block.end_sec)
    hrs = _slice_stream_by_time(time_stream, hr_stream, block.start_sec, block.end_sec)
    cads = _slice_stream_by_time(time_stream, cadence_stream, block.start_sec, block.end_sec)
    speeds = _slice_stream_by_time(time_stream, speed_stream, block.start_sec, block.end_sec)

    duration_sec = block.end_sec - block.start_sec

    avg_power = _safe_mean(watts)
    avg_hr = _safe_mean(hrs)
    avg_cadence = _safe_mean(cads) if cads else None
    avg_speed = _safe_mean(speeds) if speeds else None

    ef = None
    if avg_power is not None and avg_hr not in (None, 0):
        ef = avg_power / avg_hr

    power_cv = _safe_cv(watts)
    hr_cv = _safe_cv(hrs)
    cadence_cv = _safe_cv(cads) if cads else None
    power_variation_pct = (power_cv * 100.0) if power_cv is not None else None

    watts_halves = _split_block_stream(time_stream, watts_stream, block.start_sec, block.end_sec)
    hr_halves = _split_block_stream(time_stream, hr_stream, block.start_sec, block.end_sec)

    first_power = _safe_mean(watts_halves["first"])
    second_power = _safe_mean(watts_halves["second"])
    first_hr = _safe_mean(hr_halves["first"])
    second_hr = _safe_mean(hr_halves["second"])

    negative_split_power_pct = _pct_change(first_power, second_power)
    negative_split_hr_pct = _pct_change(first_hr, second_hr)

    ef_first = None
    ef_second = None
    if first_power is not None and first_hr not in (None, 0):
        ef_first = first_power / first_hr
    if second_power is not None and second_hr not in (None, 0):
        ef_second = second_power / second_hr

    drift_pct = _pct_change(ef_first, ef_second)
    decoupling_pct = None if drift_pct is None else -drift_pct

    stability_score = _compute_stability_score(
        duration_sec=duration_sec,
        power_variation_pct=power_variation_pct,
        decoupling_pct=decoupling_pct,
        cadence_cv=cadence_cv,
    )

    block_quality_label = _label_block_quality(stability_score, decoupling_pct)

    return BlockAnalysisResult(
        index=index,
        label=label,
        start_sec=block.start_sec,
        end_sec=block.end_sec,
        duration_sec=duration_sec,
        avg_power=_round_or_none(avg_power, 1),
        avg_hr=_round_or_none(avg_hr, 1),
        avg_cadence=_round_or_none(avg_cadence, 1),
        avg_speed_mps=_round_or_none(avg_speed, 2),
        ef=_round_or_none(ef, 3),
        power_cv=_round_or_none(power_cv, 3),
        hr_cv=_round_or_none(hr_cv, 3),
        cadence_cv=_round_or_none(cadence_cv, 3),
        power_variation_pct=_round_or_none(power_variation_pct, 1),
        negative_split_power_pct=_round_or_none(negative_split_power_pct, 1),
        negative_split_hr_pct=_round_or_none(negative_split_hr_pct, 1),
        drift_pct=_round_or_none(drift_pct, 1),
        decoupling_pct=_round_or_none(decoupling_pct, 1),
        stability_score=_round_or_none(stability_score, 1),
        block_quality_label=block_quality_label,
    )


def _compute_between_block_comparison(results: list[BlockAnalysisResult]) -> BetweenBlockComparison | None:
    if not results:
        return None

    first_power = results[0].avg_power
    first_ef = results[0].ef

    power_items: list[BlockDropItem] = []
    ef_items: list[BlockDropItem] = []

    for r in results:
        power_items.append(
            BlockDropItem(
                label=r.label,
                drop_pct=_round_or_none(_pct_change(first_power, r.avg_power), 1),
            )
        )
        ef_items.append(
            BlockDropItem(
                label=r.label,
                drop_pct=_round_or_none(_pct_change(first_ef, r.ef), 1),
            )
        )

    return BetweenBlockComparison(
        power_drop_from_first_pct=power_items,
        ef_drop_from_first_pct=ef_items,
    )


def _compute_block_summary(results: list[BlockAnalysisResult]) -> BlockSummary | None:
    if not results:
        return None

    scored = [r for r in results if r.stability_score is not None]
    if not scored:
        return BlockSummary(
            best_block_label=None,
            worst_block_label=None,
            consistency_assessment="unknown",
            overall_block_quality="unknown",
        )

    best_block = max(scored, key=lambda x: float(x.stability_score))
    worst_block = min(scored, key=lambda x: float(x.stability_score))

    scores = [float(r.stability_score) for r in scored if r.stability_score is not None]
    spread = (max(scores) - min(scores)) if scores else None
    avg_score = (sum(scores) / len(scores)) if scores else None

    if spread is None:
        consistency = "unknown"
    elif spread <= 8:
        consistency = "very_consistent"
    elif spread <= 18:
        consistency = "moderately_consistent"
    else:
        consistency = "moderate_fade"

    if avg_score is None:
        overall = "unknown"
    elif avg_score >= 85:
        overall = "good"
    elif avg_score >= 70:
        overall = "acceptable"
    elif avg_score >= 55:
        overall = "mixed"
    else:
        overall = "poor"

    return BlockSummary(
        best_block_label=best_block.label,
        worst_block_label=worst_block.label,
        consistency_assessment=consistency,
        overall_block_quality=overall,
    )


# =========================
# Basic endpoints
# =========================

@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Intervals.icu relay API is running.",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/healthz",
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# =========================
# Raw intervals passthrough
# =========================

@app.get("/intervals/activities")
def get_activities(
    start: str = Query(..., description="Start date in YYYY-MM-DD"),
    end: str = Query(..., description="End date in YYYY-MM-DD"),
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    start_d = parse_date_yyyy_mm_dd(start)
    end_d = parse_date_yyyy_mm_dd(end)
    if start_d > end_d:
        raise HTTPException(status_code=400, detail="start must be <= end")

    return JSONResponse(content=_fetch_activities(start, end))


@app.get("/intervals/wellness")
def get_wellness(
    start: str = Query(..., description="Start date in YYYY-MM-DD"),
    end: str = Query(..., description="End date in YYYY-MM-DD"),
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    start_d = parse_date_yyyy_mm_dd(start)
    end_d = parse_date_yyyy_mm_dd(end)
    if start_d > end_d:
        raise HTTPException(status_code=400, detail="start must be <= end")

    return JSONResponse(content=_fetch_wellness(start, end))


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

    r = intervals_get(f"/activity/{activity_id}/streams.json", params=params)
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
        f"/activity/{activity_id}/streams.csv",
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

    r = intervals_get(f"/activity/{activity_id}/file")
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

    r = intervals_get(f"/activity/{activity_id}/fit-file")
    return Response(
        content=r.content,
        media_type=r.headers.get("content-type", "application/octet-stream"),
        headers={
            "Content-Disposition": f'attachment; filename="{activity_id}.fit.gz"',
        },
    )


# =========================
# Analysis endpoints
# =========================

@app.get("/analysis/training-quality")
def analyze_training_quality(
    start: str = Query(..., description="Start date in YYYY-MM-DD"),
    end: str = Query(..., description="End date in YYYY-MM-DD"),
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    start_d = parse_date_yyyy_mm_dd(start)
    end_d = parse_date_yyyy_mm_dd(end)
    if start_d > end_d:
        raise HTTPException(status_code=400, detail="start must be <= end")

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
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    start_d = parse_date_yyyy_mm_dd(start)
    end_d = parse_date_yyyy_mm_dd(end)
    if start_d > end_d:
        raise HTTPException(status_code=400, detail="start must be <= end")

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
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    start_d = parse_date_yyyy_mm_dd(start)
    end_d = parse_date_yyyy_mm_dd(end)
    if start_d > end_d:
        raise HTTPException(status_code=400, detail="start must be <= end")

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
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    granularity: str = Query(default="week", pattern="^(day|week)$"),
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    start_d = parse_date_yyyy_mm_dd(start)
    end_d = parse_date_yyyy_mm_dd(end)
    if start_d > end_d:
        raise HTTPException(status_code=400, detail="start must be <= end")

    activities = _fetch_activities(start, end)
    wellness = _fetch_wellness(start, end)

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
        for key in ["date", "day", "wellness_date", "id", "updated"]:
            val = w.get(key)
            if isinstance(val, str) and len(val) >= 10:
                day_str = val[:10]
                break
        if not day_str:
            continue

        try:
            d = parse_date_yyyy_mm_dd(day_str)
        except HTTPException:
            continue

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
# Detailed single-workout analysis
# =========================

@app.get("/analysis/workout-detail/{activity_id}")
def analyze_workout_detail(
    activity_id: str,
    warmup_exclude_seconds: int = Query(default=600, ge=0, le=3600),
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    r = intervals_get(
        f"/activity/{activity_id}/streams.json",
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


@app.get("/analysis/endurance-efficiency-detail")
def analyze_endurance_efficiency_detail(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    start_d = parse_date_yyyy_mm_dd(start)
    end_d = parse_date_yyyy_mm_dd(end)
    if start_d > end_d:
        raise HTTPException(status_code=400, detail="start must be <= end")

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


# =========================
# New: block-level workout analysis
# =========================

@app.post("/analysis/workout-block-quality", response_model=WorkoutBlockQualityResponse)
def analyze_workout_block_quality(
    req: WorkoutBlockQualityRequest,
    authorization: str | None = Header(default=None),
):
    verify_bearer(authorization)

    sorted_blocks = sorted(req.blocks, key=lambda b: b.start_sec)

    for i in range(1, len(sorted_blocks)):
        prev = sorted_blocks[i - 1]
        curr = sorted_blocks[i]
        if curr.start_sec < prev.end_sec:
            raise HTTPException(
                status_code=400,
                detail=f"Blocks overlap: '{prev.label or f'block_{i}'}' and '{curr.label or f'block_{i+1}'}'",
            )

    r = intervals_get(
        f"/activity/{req.activity_id}/streams.json",
        params={"types": "time,watts,heartrate,cadence,velocity_smooth"},
    )
    payload = r.json()
    smap = _extract_stream_map(payload)

    available_streams = sorted([k for k, v in smap.items() if v])

    missing_required = [s for s in REQUIRED_BLOCK_STREAMS if not smap.get(s)]
    if missing_required:
        return WorkoutBlockQualityResponse(
            activity_id=req.activity_id,
            analysis_basis="unavailable",
            available_streams=available_streams,
            block_count=len(sorted_blocks),
            blocks_analyzed=[],
            between_block_comparison=None,
            summary=None,
            limitations=[
                "Required streams for block analysis are missing.",
                f"Missing required streams: {', '.join(sorted(missing_required))}",
                "Detailed block analysis requires at least time, watts, and heartrate.",
                "Block boundaries are user-specified and not auto-detected.",
            ],
        )

    time_stream = smap.get("time", [])
    if not time_stream:
        return WorkoutBlockQualityResponse(
            activity_id=req.activity_id,
            analysis_basis="unavailable",
            available_streams=available_streams,
            block_count=len(sorted_blocks),
            blocks_analyzed=[],
            between_block_comparison=None,
            summary=None,
            limitations=[
                "time stream is empty.",
                "Detailed block analysis requires non-empty time, watts, and heartrate streams.",
            ],
        )

    max_time = max([t for t in time_stream if not math.isnan(t)], default=None)
    if max_time is None:
        return WorkoutBlockQualityResponse(
            activity_id=req.activity_id,
            analysis_basis="unavailable",
            available_streams=available_streams,
            block_count=len(sorted_blocks),
            blocks_analyzed=[],
            between_block_comparison=None,
            summary=None,
            limitations=[
                "time stream contains no valid values.",
                "Detailed block analysis requires valid time, watts, and heartrate streams.",
            ],
        )

    for block in sorted_blocks:
        if block.start_sec > max_time:
            raise HTTPException(
                status_code=400,
                detail=f"Block start_sec {block.start_sec} exceeds activity duration",
            )

    results: list[BlockAnalysisResult] = []
    for idx, block in enumerate(sorted_blocks):
        results.append(_analyze_single_block(idx, block, smap))

    between = _compute_between_block_comparison(results)
    summary = _compute_block_summary(results)

    return WorkoutBlockQualityResponse(
        activity_id=req.activity_id,
        analysis_basis="streams",
        available_streams=available_streams,
        block_count=len(sorted_blocks),
        blocks_analyzed=results,
        between_block_comparison=between,
        summary=summary,
        limitations=[
            "Block boundaries are user-specified and not auto-detected.",
            "EF is calculated here as average power divided by average heart rate.",
            "Drift/decoupling is estimated from first-half vs second-half block comparison.",
            "This endpoint evaluates each block individually but does not infer workout intent automatically.",
        ],
    )
