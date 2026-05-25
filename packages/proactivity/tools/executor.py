"""Tool definitions for the proactivity agent's Claude calls.

Two flavors of tool:

- **Server-side** (web_search): declared in `_WEB_SEARCH_TOOL`. Anthropic
  executes the tool round-trip inside the `messages.create` call. We
  see `server_tool_use` + `web_search_tool_result` blocks in the
  response and a final `text` block. `stop_reason` is usually
  `end_turn`; on the rare server-side iteration cap we get `pause_turn`
  and resend.

- **Client-side** (Google Maps + Google Calendar): declared in
  `_CUSTOM_TOOLS`. Claude emits `tool_use` blocks with
  `stop_reason == "tool_use"` and our agentic loop in
  `agent._call_claude` calls `execute_tool()` to run them. Maps tools
  hit Google's HTTP APIs with `GOOGLE_MAPS_API_KEY` from `.env`.
  Calendar tools use OAuth: a long-lived refresh token in `.env`
  (generated once via `proactivity/calendar_auth.py`) is exchanged
  for a short-lived access token on demand and cached in-process.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from reflections.config import (
    DEFAULT_LOCATION_LAT,
    DEFAULT_LOCATION_LON,
    DEFAULT_LOCATION_NAME,
    DEFAULT_LOCATION_RADIUS_M,
)

from proactivity.promptlog import log_event

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_S = 5.0


_WEB_SEARCH_TOOL: dict[str, Any] = {
    # Older fully-server-side variant. The newer 20260209 variant
    # defaults to programmatic mode which Haiku 4.5 doesn't support.
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 3,
}


_CUSTOM_TOOLS: list[dict[str, Any]] = [
    {
        "name": "places_search",
        "description": (
            "Find places near the wearer (default location from config) that "
            "match a free-text query. Use this for 'find me X', 'where is "
            "a good X', 'what's a nearby X' style questions. Returns the "
            "top 5 places, each with a place_id you can pass to "
            "place_details for follow-up."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to search for, e.g. 'sushi restaurants', "
                        "'coffee shops', '24 hour pharmacy'."
                    ),
                },
                "location_bias_radius_m": {
                    "type": "integer",
                    "description": (
                        "Radius in meters around the default location to bias the "
                        "search. Default 5000. Increase for broader "
                        "searches (e.g. 20000 for a wider area)."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "place_details",
        "description": (
            "Look up full details for a specific place by place_id. Use "
            "this after places_search when the user asks 'is it open?', "
            "'what's their phone number?', 'how is it rated?'. The "
            "place_id comes from a prior places_search result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "place_id": {
                    "type": "string",
                    "description": (
                        "Google Maps place_id (e.g. 'ChIJ...'). Take "
                        "this from a previous places_search response."
                    ),
                },
            },
            "required": ["place_id"],
        },
    },
    {
        "name": "directions",
        "description": (
            "Get directions from one place to another. Use for 'how do "
            "I get to X', 'how long to walk to X', 'directions to X'. "
            "Origin defaults to the wearer's configured location when not specified."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": (
                        "Where the user wants to go. Place name or "
                        "address, e.g. 'City Hall', '123 Main St'."
                    ),
                },
                "origin": {
                    "type": "string",
                    "description": (
                        "Starting point. Defaults to the configured "
                        "DEFAULT_LOCATION_NAME when omitted. Pass an address "
                        "or place name."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["walking", "driving", "bicycling", "transit"],
                    "description": (
                        "Travel mode. Default 'walking' (most common for "
                        "the smart-glasses use case)."
                    ),
                },
            },
            "required": ["destination"],
        },
    },
    {
        "name": "list_calendar_events",
        "description": (
            "List the wearer's upcoming Google Calendar events. Use for "
            "'what's on my calendar', 'what do I have today', 'am I free "
            "this afternoon', 'what's next'. Returns up to 10 events "
            "ordered by start time. Defaults to the next 7 days from now "
            "if no time bounds are given."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "time_min": {
                    "type": "string",
                    "description": (
                        "RFC3339 lower bound on event end time, e.g. "
                        "'2026-05-04T00:00:00-07:00'. Defaults to now."
                    ),
                },
                "time_max": {
                    "type": "string",
                    "description": (
                        "RFC3339 upper bound on event start time. "
                        "Defaults to 7 days after time_min."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "How many events to return (1-10). Default 10.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "find_calendar_event",
        "description": (
            "Search the wearer's Google Calendar for events matching a "
            "free-text query — title, description, location, attendee. "
            "Use for 'when's my dentist appt', 'do I have a meeting with "
            "Sarah this week', 'what time is the standup'. Returns up "
            "to 5 matching events sorted by start time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to search for, e.g. 'dentist', 'standup', " "'lunch with Sarah'."
                    ),
                },
                "time_min": {
                    "type": "string",
                    "description": (
                        "Optional RFC3339 lower bound. Defaults to now "
                        "(only future/in-progress events)."
                    ),
                },
                "time_max": {
                    "type": "string",
                    "description": (
                        "Optional RFC3339 upper bound. Defaults to 30 " "days after time_min."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_calendar_event",
        "description": (
            "Create a new event on the wearer's primary Google Calendar. "
            "Use for 'add X to my calendar', 'schedule a meeting with Y', "
            "'remind me to Z at 3pm'. The wearer's local timezone is "
            "America/Los_Angeles — interpret loose times "
            "('tomorrow at 3') in that zone. Pass attendee emails only "
            "when the user named someone whose email you actually know. "
            "Calendar invites are NOT emailed by default — only set "
            "notify_attendees=true when the user explicitly asked to "
            "send the invite."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Event title (the 'summary' field).",
                },
                "start": {
                    "type": "string",
                    "description": (
                        "Start time. RFC3339 with timezone offset for "
                        "timed events ('2026-05-05T15:00:00-07:00'), or "
                        "YYYY-MM-DD for all-day events."
                    ),
                },
                "end": {
                    "type": "string",
                    "description": (
                        "End time, same format as start. Optional — "
                        "defaults to start + 1 hour for timed events, "
                        "or the same day for all-day events."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Optional longer description / notes.",
                },
                "location": {
                    "type": "string",
                    "description": "Optional location string.",
                },
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of attendee email addresses. "
                        "Only include emails the user explicitly named."
                    ),
                },
                "notify_attendees": {
                    "type": "boolean",
                    "description": (
                        "Email the invite to attendees. Default false. "
                        "Set true only when the user asked to send it."
                    ),
                },
            },
            "required": ["title", "start"],
        },
    },
]


def build_anthropic_tools() -> list[dict[str, Any]]:
    """Tool definitions for `messages.create(tools=...)`.

    Stable across calls so the prompt cache stays warm — tools render
    at position 0 of the cacheable prefix; reordering invalidates the
    cache."""
    return [_WEB_SEARCH_TOOL, *_CUSTOM_TOOLS]


def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    """Execute a client-side tool and return a compact text result.

    Errors are returned as `'Tool error: ...'` strings rather than
    raised — `_call_claude`'s loop already routes any return value
    into a `tool_result` block, so the model can recover gracefully."""
    log_event("tool", "call", {"name": name, "input": tool_input})
    if name == "places_search":
        result = _places_search(tool_input)
    elif name == "place_details":
        result = _place_details(tool_input)
    elif name == "directions":
        result = _directions(tool_input)
    elif name == "list_calendar_events":
        result = _list_calendar_events(tool_input)
    elif name == "find_calendar_event":
        result = _find_calendar_event(tool_input)
    elif name == "create_calendar_event":
        result = _create_calendar_event(tool_input)
    else:
        result = f"Tool error: unknown tool {name!r}"
    log_event("tool", "result", {"name": name, "output": result})
    return result


# ---------------------------------------------------- google maps helpers


def _api_key() -> str | None:
    return os.environ.get("GOOGLE_MAPS_API_KEY") or None


def _http_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> dict[str, Any]:
    """Fetch JSON with a hard timeout. Raises urllib.error.URLError or
    json.JSONDecodeError on failure; callers convert to 'Tool error: ...'."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _places_search(tool_input: dict[str, Any]) -> str:
    key = _api_key()
    if not key:
        return "Tool error: GOOGLE_MAPS_API_KEY not set"

    query = (tool_input.get("query") or "").strip()
    if not query:
        return "Tool error: query is required"
    radius = int(tool_input.get("location_bias_radius_m") or DEFAULT_LOCATION_RADIUS_M)

    # Places API (New) — Text Search. Field mask trims the response and
    # also caps billing tier (we stay within Pro SKU by listing only
    # Pro+Basic fields).
    url = "https://places.googleapis.com/v1/places:searchText"
    body = {
        "textQuery": query,
        "locationBias": {
            "circle": {
                "center": {
                    "latitude": DEFAULT_LOCATION_LAT,
                    "longitude": DEFAULT_LOCATION_LON,
                },
                "radius": float(radius),
            }
        },
        "maxResultCount": 5,
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.formattedAddress,"
            "places.rating,places.userRatingCount,"
            "places.currentOpeningHours.openNow"
        ),
    }
    try:
        data = _http_json(
            url,
            method="POST",
            headers=headers,
            body=json.dumps(body).encode("utf-8"),
        )
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            err_body = ""
        return f"Tool error: places_search HTTP {e.code} — {err_body}"
    except Exception as e:
        return f"Tool error: places_search failed — {e}"

    places = data.get("places") or []
    if not places:
        return f"No places found for {query!r} near {DEFAULT_LOCATION_NAME}."

    lines = [f"Top results for {query!r}:"]
    for i, p in enumerate(places[:5], 1):
        name = (p.get("displayName") or {}).get("text") or "(unknown)"
        addr = p.get("formattedAddress") or ""
        rating = p.get("rating")
        n = p.get("userRatingCount")
        rating_str = f" — {rating}★ ({n})" if rating and n else f" — {rating}★" if rating else ""
        open_now = (p.get("currentOpeningHours") or {}).get("openNow")
        open_str = (
            " — open now" if open_now is True else " — closed now" if open_now is False else ""
        )
        pid = p.get("id") or ""
        lines.append(f"{i}. {name}{rating_str}{open_str} — {addr} " f"[place_id={pid}]")
    return "\n".join(lines)


def _place_details(tool_input: dict[str, Any]) -> str:
    key = _api_key()
    if not key:
        return "Tool error: GOOGLE_MAPS_API_KEY not set"

    pid = (tool_input.get("place_id") or "").strip()
    if not pid:
        return "Tool error: place_id is required"

    url = f"https://places.googleapis.com/v1/places/{urllib.parse.quote(pid)}"
    headers = {
        "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": (
            "id,displayName,formattedAddress,rating,userRatingCount,"
            "nationalPhoneNumber,websiteUri,"
            "currentOpeningHours.openNow,"
            "currentOpeningHours.weekdayDescriptions"
        ),
    }
    try:
        p = _http_json(url, headers=headers)
    except urllib.error.HTTPError as e:
        return f"Tool error: place_details HTTP {e.code}"
    except Exception as e:
        return f"Tool error: place_details failed — {e}"

    name = (p.get("displayName") or {}).get("text") or "(unknown)"
    addr = p.get("formattedAddress") or ""
    rating = p.get("rating")
    n = p.get("userRatingCount")
    phone = p.get("nationalPhoneNumber") or ""
    site = p.get("websiteUri") or ""
    hours = p.get("currentOpeningHours") or {}
    open_now = hours.get("openNow")
    today_hours = ""
    weekdays = hours.get("weekdayDescriptions") or []
    if weekdays:
        # weekdayDescriptions is a 7-element list starting at Monday;
        # keep just today's entry to stay terse.
        import datetime

        idx = datetime.datetime.now().weekday()
        if 0 <= idx < len(weekdays):
            today_hours = weekdays[idx]

    parts = [name]
    if addr:
        parts.append(addr)
    if rating and n:
        parts.append(f"{rating}★ ({n} reviews)")
    elif rating:
        parts.append(f"{rating}★")
    if open_now is True:
        parts.append("open now")
    elif open_now is False:
        parts.append("closed now")
    if today_hours:
        parts.append(f"Today: {today_hours}")
    if phone:
        parts.append(f"Phone: {phone}")
    if site:
        parts.append(f"Web: {site}")
    return ". ".join(parts) + "."


def _directions(tool_input: dict[str, Any]) -> str:
    key = _api_key()
    if not key:
        return "Tool error: GOOGLE_MAPS_API_KEY not set"

    destination = (tool_input.get("destination") or "").strip()
    if not destination:
        return "Tool error: destination is required"
    origin = (tool_input.get("origin") or "").strip() or DEFAULT_LOCATION_NAME
    mode = (tool_input.get("mode") or "walking").strip().lower()
    if mode not in ("walking", "driving", "bicycling", "transit"):
        mode = "walking"

    params = urllib.parse.urlencode(
        {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "key": key,
        }
    )
    url = f"https://maps.googleapis.com/maps/api/directions/json?{params}"
    try:
        data = _http_json(url)
    except urllib.error.HTTPError as e:
        return f"Tool error: directions HTTP {e.code}"
    except Exception as e:
        return f"Tool error: directions failed — {e}"

    status = data.get("status")
    if status != "OK":
        msg = data.get("error_message") or status or "unknown"
        return f"Tool error: directions {status} — {msg}"

    routes = data.get("routes") or []
    if not routes:
        return f"No route found from {origin!r} to {destination!r}."

    leg = (routes[0].get("legs") or [{}])[0]
    duration = (leg.get("duration") or {}).get("text") or "?"
    distance = (leg.get("distance") or {}).get("text") or "?"
    steps = leg.get("steps") or []

    # Strip HTML tags from the step instructions; cap to first 3 to
    # keep the response compact for Claude.
    import re

    def _clean(s: str) -> str:
        return re.sub(r"<[^>]+>", "", s).strip()

    step_lines = []
    for s in steps[:3]:
        instr = _clean(s.get("html_instructions") or "")
        sd = (s.get("distance") or {}).get("text") or ""
        if instr:
            step_lines.append(f"  - {instr} ({sd})" if sd else f"  - {instr}")

    out = [f"{duration} by {mode}, {distance} (from {origin!r} to " f"{destination!r})."]
    if step_lines:
        out.append("First steps:")
        out.extend(step_lines)
    return "\n".join(out)


# ------------------------------------------------- google calendar helpers
#
# Auth model: a one-time consent flow (proactivity/calendar_auth.py)
# generates a refresh token. The runtime trades that for a short-lived
# access token via the standard OAuth refresh grant and caches it
# in-process until ~60 s before expiry. No google-api-python-client
# dependency — REST + urllib is enough for the three operations we
# need (list, search, insert).

_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
_CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"
_LOCAL_TZ = "America/Los_Angeles"

_calendar_token_lock = threading.Lock()
_calendar_token_cache: dict[str, Any] = {"access_token": None, "expires_at": 0.0}


def _calendar_oauth_config() -> tuple[str, str, str] | None:
    cid = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
    sec = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
    rt = os.environ.get("GOOGLE_OAUTH_REFRESH_TOKEN")
    if cid and sec and rt:
        return cid, sec, rt
    return None


def _calendar_access_token() -> str | None:
    """Return a cached access token, refreshing via the OAuth refresh
    grant when expired. Returns None if env credentials are missing."""
    cfg = _calendar_oauth_config()
    if cfg is None:
        return None
    cid, sec, refresh = cfg

    with _calendar_token_lock:
        now = time.time()
        cached = _calendar_token_cache.get("access_token")
        if cached and now < _calendar_token_cache["expires_at"]:
            return cached

        body = urllib.parse.urlencode(
            {
                "client_id": cid,
                "client_secret": sec,
                "refresh_token": refresh,
                "grant_type": "refresh_token",
            }
        ).encode("utf-8")
        try:
            data = _http_json(
                _OAUTH_TOKEN_URL,
                method="POST",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                body=body,
            )
        except Exception as e:
            logger.error("[tools] calendar token refresh failed: %s", e)
            return None
        token = data.get("access_token")
        if not token:
            return None
        # 60 s buffer so we don't race the server's expiry clock.
        ttl = int(data.get("expires_in") or 3600) - 60
        _calendar_token_cache["access_token"] = token
        _calendar_token_cache["expires_at"] = now + max(ttl, 30)
        return token


def _calendar_request(
    path: str,
    *,
    method: str = "GET",
    query: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Authed Calendar API call. Caller wraps for tool-friendly errors."""
    token = _calendar_access_token()
    if not token:
        raise RuntimeError("calendar credentials not configured")
    url = f"{_CALENDAR_API_BASE}{path}"
    if query:
        # Drop None values so callers can pass optional params uniformly.
        cleaned = {k: v for k, v in query.items() if v is not None}
        if cleaned:
            url = f"{url}?{urllib.parse.urlencode(cleaned, doseq=True)}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    raw_body: bytes | None = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        raw_body = json.dumps(body).encode("utf-8")
    return _http_json(url, method=method, headers=headers, body=raw_body)


def _now_rfc3339() -> str:
    # Use UTC with offset; Calendar accepts any RFC3339 with offset.
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _shift_rfc3339(base: str, days: int) -> str:
    """Add `days` to an RFC3339 timestamp. Used to default time_max
    relative to time_min without pulling in dateutil."""
    from datetime import datetime, timedelta, timezone

    s = base.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.now(timezone.utc)
    return (dt + timedelta(days=days)).isoformat()


def _format_event(ev: dict[str, Any]) -> str:
    title = ev.get("summary") or "(no title)"
    start = ev.get("start") or {}
    end = ev.get("end") or {}
    when_start = start.get("dateTime") or start.get("date") or "?"
    when_end = end.get("dateTime") or end.get("date") or ""
    loc = ev.get("location") or ""
    parts = [f"{title} — {when_start}"]
    if when_end:
        parts[0] += f" → {when_end}"
    if loc:
        parts.append(f"@ {loc}")
    attendees = ev.get("attendees") or []
    names = [a.get("displayName") or a.get("email") or "?" for a in attendees if not a.get("self")]
    if names:
        parts.append("with " + ", ".join(names[:5]))
    return " ".join(parts)


def _calendar_credentials_error() -> str:
    return (
        "Tool error: GOOGLE_OAUTH_CLIENT_ID/SECRET/REFRESH_TOKEN not set. "
        "Run `python -m proactivity.calendar_auth` once to generate them."
    )


def _list_calendar_events(tool_input: dict[str, Any]) -> str:
    if _calendar_oauth_config() is None:
        return _calendar_credentials_error()

    time_min = (tool_input.get("time_min") or "").strip() or _now_rfc3339()
    time_max = (tool_input.get("time_max") or "").strip() or _shift_rfc3339(time_min, 7)
    try:
        max_results = int(tool_input.get("max_results") or 10)
    except (TypeError, ValueError):
        max_results = 10
    max_results = max(1, min(max_results, 10))

    try:
        data = _calendar_request(
            "/calendars/primary/events",
            query={
                "timeMin": time_min,
                "timeMax": time_max,
                "maxResults": max_results,
                "singleEvents": "true",
                "orderBy": "startTime",
                "timeZone": _LOCAL_TZ,
            },
        )
    except urllib.error.HTTPError as e:
        return f"Tool error: list_calendar_events HTTP {e.code}"
    except Exception as e:
        return f"Tool error: list_calendar_events failed — {e}"

    items = data.get("items") or []
    if not items:
        return f"No events between {time_min} and {time_max}."
    lines = [f"Events {time_min} → {time_max}:"]
    for i, ev in enumerate(items, 1):
        lines.append(f"{i}. {_format_event(ev)}")
    return "\n".join(lines)


def _find_calendar_event(tool_input: dict[str, Any]) -> str:
    if _calendar_oauth_config() is None:
        return _calendar_credentials_error()

    query = (tool_input.get("query") or "").strip()
    if not query:
        return "Tool error: query is required"
    time_min = (tool_input.get("time_min") or "").strip() or _now_rfc3339()
    time_max = (tool_input.get("time_max") or "").strip() or _shift_rfc3339(time_min, 30)

    try:
        data = _calendar_request(
            "/calendars/primary/events",
            query={
                "q": query,
                "timeMin": time_min,
                "timeMax": time_max,
                "maxResults": 5,
                "singleEvents": "true",
                "orderBy": "startTime",
                "timeZone": _LOCAL_TZ,
            },
        )
    except urllib.error.HTTPError as e:
        return f"Tool error: find_calendar_event HTTP {e.code}"
    except Exception as e:
        return f"Tool error: find_calendar_event failed — {e}"

    items = data.get("items") or []
    if not items:
        return f"No events matched {query!r} in the next 30 days."
    lines = [f"Matches for {query!r}:"]
    for i, ev in enumerate(items, 1):
        lines.append(f"{i}. {_format_event(ev)}")
    return "\n".join(lines)


def _looks_all_day(s: str) -> bool:
    """A 'YYYY-MM-DD' string with no time component → all-day event."""
    return len(s) == 10 and s[4] == "-" and s[7] == "-"


def _create_calendar_event(tool_input: dict[str, Any]) -> str:
    if _calendar_oauth_config() is None:
        return _calendar_credentials_error()

    title = (tool_input.get("title") or "").strip()
    start = (tool_input.get("start") or "").strip()
    if not title or not start:
        return "Tool error: title and start are required"
    end = (tool_input.get("end") or "").strip()
    description = (tool_input.get("description") or "").strip()
    location = (tool_input.get("location") or "").strip()
    attendees_raw = tool_input.get("attendees") or []
    notify = bool(tool_input.get("notify_attendees"))

    all_day = _looks_all_day(start)
    if not end:
        if all_day:
            end = start
        else:
            # Default timed events to start + 1h.
            from datetime import datetime, timedelta

            try:
                dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                end = (dt + timedelta(hours=1)).isoformat()
            except ValueError:
                return f"Tool error: could not parse start={start!r}"

    if all_day != _looks_all_day(end):
        return "Tool error: start and end must both be all-day or both timed"

    body: dict[str, Any] = {"summary": title}
    if all_day:
        body["start"] = {"date": start}
        body["end"] = {"date": end}
    else:
        body["start"] = {"dateTime": start, "timeZone": _LOCAL_TZ}
        body["end"] = {"dateTime": end, "timeZone": _LOCAL_TZ}
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    if isinstance(attendees_raw, list) and attendees_raw:
        body["attendees"] = [{"email": str(a)} for a in attendees_raw if str(a).strip()]

    try:
        data = _calendar_request(
            "/calendars/primary/events",
            method="POST",
            query={"sendUpdates": "all" if notify else "none"},
            body=body,
        )
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            err_body = ""
        return f"Tool error: create_calendar_event HTTP {e.code} — {err_body}"
    except Exception as e:
        return f"Tool error: create_calendar_event failed — {e}"

    summary = data.get("summary") or title
    start_obj = data.get("start") or {}
    when = start_obj.get("dateTime") or start_obj.get("date") or start
    link = data.get("htmlLink") or ""
    out = f"Created: {summary} at {when}"
    if data.get("attendees"):
        n = len([a for a in data["attendees"] if not a.get("self")])
        if n:
            out += f" with {n} attendee(s){' (notified)' if notify else ' (not notified)'}"
    if link:
        out += f" — {link}"
    return out
