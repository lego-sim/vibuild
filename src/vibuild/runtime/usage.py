"""Read-only per-request token accounting from Codex rollout logs.

Example: vibuild tools usage --session main=THREAD_ID --session builder=THREAD_ID
  --start ISO_OR_EPOCH --end ISO_OR_EPOCH --output REPORT_DIRECTORY
Roles and time windows are explicit attribution, not exact semantic task accounting.
"""

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path

FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def timestamp(value):
    try:
        return float(value)
    except ValueError:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def aggregate(records):
    totals = {key: sum(row["usage"].get(key, 0) for row in records) for key in FIELDS}
    totals["uncached_input_tokens"] = (
        totals["input_tokens"] - totals["cached_input_tokens"]
    )
    totals["requests"] = len(records)
    totals["mean_input_per_request"] = (
        round(totals["input_tokens"] / len(records), 1) if records else 0
    )
    totals["max_input_per_request"] = max(
        (r["usage"].get("input_tokens", 0) for r in records), default=0
    )
    return totals


def read_usage(path, thread_id, start, end, seen):
    rows = []
    warnings = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
            except ValueError:
                warnings.append(
                    f"{path.name}:{number}: incomplete/invalid JSON skipped"
                )
                continue
            if event.get("type") != "token_usage_record":
                continue  # Never add cumulative token_count mirrors.
            payload = event.get("payload", {})
            if payload.get("thread_id") != thread_id:
                continue  # Inherited parent records are not child consumption.
            when = timestamp(event["timestamp"])
            if not start <= when < end:
                continue
            response = payload.get("response_id")
            if not response:
                warnings.append(
                    f"{path.name}:{number}: missing response_id; excluded to avoid double counting"
                )
                continue
            if response in seen:
                continue
            usage = payload.get("usage")
            if not isinstance(usage, dict) or not all(key in usage for key in FIELDS):
                warnings.append(f"{path.name}:{number}: incomplete usage; excluded")
                continue
            seen.add(response)
            rows.append(
                {
                    "timestamp": event["timestamp"],
                    "thread_id": thread_id,
                    "turn_id": payload.get("turn_id"),
                    "response_id": response,
                    "usage": {key: usage[key] for key in FIELDS},
                }
            )
    return rows, warnings


def build_report(sessions, logs, start, end):
    seen = set()
    roles = []
    warnings = []
    all_rows = []
    for role, thread in sessions:
        files = sorted(logs.rglob(f"*{thread}*.jsonl"))
        if not files:
            warnings.append(f"No rollout for {role}: {thread}")
        rows = []
        for path in files:
            found, notes = read_usage(path, thread, start, end, seen)
            rows.extend(found)
            warnings.extend(notes)
        turns = []
        for turn in dict.fromkeys(row["turn_id"] for row in rows):
            turns.append(
                {
                    "turn_id": turn,
                    **aggregate([r for r in rows if r["turn_id"] == turn]),
                }
            )
        roles.append(
            {
                "role": role,
                "thread_id": thread,
                "sources": [str(p) for p in files],
                **aggregate(rows),
                "turns": turns,
            }
        )
        all_rows.extend(rows)
    return {
        "window": {
            "start_inclusive": datetime.fromtimestamp(start, timezone.utc).isoformat(),
            "end_exclusive": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        },
        "method": "Sum deduplicated per-response usage for explicitly selected thread IDs and completion timestamps.",
        "limitations": [
            "Role labels and time window are attribution assumptions; shared main activity may include other regions.",
            "Requests crossing a boundary are assigned by usage-record timestamp; not prorated.",
            "Cached input is a subset of input; reasoning output is a subset of output. Do not add them twice.",
            "Totals include repeated cached context; these are not prices, bills, or unique content tokens.",
            "No exact split into images, prompts, tool results, or semantic reasoning tasks is provided by these usage fields.",
            "Missing records and unfinished responses are not estimated; tools running locally have no separate model token charge in this report.",
        ],
        "totals": aggregate(all_rows),
        "roles": roles,
        "warnings": warnings,
        "requests": sorted(all_rows, key=lambda row: row["timestamp"]),
    }


def save_report(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = [
        "role",
        "requests",
        "total_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "mean_input_per_request",
    ]
    rows = report["roles"] + [{"role": "TOTAL", **report["totals"]}]
    table = (
        "<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr>"
    )
    for row in rows:
        table += (
            "<tr>"
            + "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in columns)
            + "</tr>"
        )
    table += "</table>"
    body = '<!doctype html><meta charset="utf-8"><title>Iteration token accounting</title><style>body{font:16px system-ui;margin:32px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc;text-align:right}li{margin:8px}</style><h1>Iteration token accounting</h1>'
    body += (
        "<p>"
        + html.escape(str(report["window"]))
        + "</p>"
        + table
        + "<h2>Method and limits</h2><p>"
        + html.escape(report["method"])
        + "</p><ul>"
    )
    body += (
        "".join(
            "<li>" + html.escape(s) + "</li>"
            for s in report["limitations"] + report["warnings"]
        )
        + "</ul>"
    )
    (output / "report.html").write_text(body)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session", action="append", required=True, metavar="ROLE=THREAD_ID"
    )
    parser.add_argument("--logs", type=Path, default=Path.home() / ".codex/sessions")
    parser.add_argument("--start", required=True, type=timestamp)
    parser.add_argument("--end", required=True, type=timestamp)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.end <= args.start:
        parser.error("end must be after start")
    sessions = [item.split("=", 1) for item in args.session]
    if any(len(item) != 2 or not all(item) for item in sessions):
        parser.error("session must be ROLE=THREAD_ID")
    if len({t for _, t in sessions}) != len(sessions):
        parser.error("Each thread must have exactly one role")
    report = build_report(sessions, args.logs, args.start, args.end)
    save_report(report, args.output)
    print(
        json.dumps(
            {k: report[k] for k in ("window", "totals", "roles", "warnings")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
