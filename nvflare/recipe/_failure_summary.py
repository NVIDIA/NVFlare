# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded failure presentation from existing job logs, without changing run status."""

import json
import re
import tempfile
from itertools import islice
from pathlib import Path

from nvflare.apis.fl_constant import ReturnCode
from nvflare.fuel.utils.log_utils import _read_log_tail, wrap_log_message

_MAX_LOG_BYTES = 1024 * 1024
_MAX_LOG_FILES = 20
_FRAME = re.compile(r'^  File "([^"\n]+)", line (\d+), in ([^\n]+)', re.MULTILINE)
_EXCEPTION = re.compile(r"^[A-Za-z_][\w.]*(?::[^\n]*)?$", re.MULTILINE)
_TEXT_ERROR = re.compile(
    r"^\d{4}-\d\d-\d\d [\d:,.]+ - (.*?) - (ERROR|CRITICAL) - (.*?)" r"(?=^\d{4}-\d\d-\d\d [\d:,.]+ - |\Z)",
    re.MULTILINE | re.DOTALL,
)


def collect_client_errors(session, job_id, result):
    """Retain existing, site-authorized error streams alongside a failed result.

    Enumerate stored error-log components before requesting at most 20 client
    tails of 1 MiB each, bounded on the server before transfer. No new streaming
    is enabled and no client is contacted here.
    """
    components = session.list_job_components(job_id) or []
    sites = dict.fromkeys(c.removeprefix("ERRORLOG_") for c in components if c.startswith("ERRORLOG_"))
    folder = None
    for site in islice(sites, _MAX_LOG_FILES):
        # The existing protocol treats these case-insensitively as selectors,
        # even when a stored component belongs to a client with that name.
        if site.lower() in ("all", "server") or not re.fullmatch(r"[\w.-]+", site) or site in (".", ".."):
            continue
        content = (
            session.get_job_logs(job_id, target=site, log_file_name="error_log.txt", max_bytes=_MAX_LOG_BYTES)
            .get("logs", {})
            .get(site)
        )
        if not content:
            continue
        if folder is None:
            folder = Path(tempfile.mkdtemp(prefix="failure-logs-", dir=result))
        target = folder / site
        target.mkdir()
        data = content.encode("utf-8")
        if len(data) > _MAX_LOG_BYTES:
            data = data[-_MAX_LOG_BYTES:].partition(b"\n")[2]
        (target / "error_log.txt").write_bytes(data)


def _records(data, plain_text):
    if plain_text:
        for match in _TEXT_ERROR.finditer(data.decode("utf-8", errors="replace")):
            # Text formatters render FL context before the message; JSON keeps
            # it in a separate field. Preserve that field for peer attribution.
            context = re.match(r"^(\[\w+=[^\n]*?\])(?: - |: )(.*)", match[3], re.DOTALL)
            yield dict(
                name=match[1],
                levelname=match[2],
                message=context[2] if context else match[3],
                fl_ctx=context[1] if context else "",
            )
    else:
        for raw in data.splitlines():
            try:
                yield json.loads(raw)
            except (ValueError, UnicodeError):
                continue


def _display(value, limit=512):
    text = str(value)
    text = text[:limit] + ("..." if len(text) > limit else "")
    return json.dumps(text, ensure_ascii=True)[1:-1]


def _first_error(path):
    """Read the first ERROR in a bounded recent snapshot of one site's log."""
    with path.open("rb") as stream:
        data, _ = _read_log_tail(stream, _MAX_LOG_BYTES, whole_lines=True)
    for record in _records(data, path.name == "error_log.txt"):
        if not isinstance(record, dict) or record.get("levelname") not in ("ERROR", "CRITICAL"):
            continue
        message = record.get("message")
        if not isinstance(message, str) or not message.strip():
            continue
        location = ""
        application_frame = False
        if "Traceback (most recent call last):" in message:
            frame_matches = list(_FRAME.finditer(message))
            frames = [frame.groups() for frame in frame_matches]
            # The exception header follows the final frame. Later unindented lines
            # can be message continuations or notes, not additional exceptions.
            exception = _EXCEPTION.search(message, frame_matches[-1].end() if frame_matches else 0)
            custom_frames = [frame for frame in frames if "/custom/" in frame[0].replace("\\", "/")]
            application_frame = bool(custom_frames)
            if custom_frames or frames:
                file, line, function = (custom_frames or frames)[-1]
                location = f"{file.replace(chr(92), '/').rsplit('/', 1)[-1]}:{line} ({function})"
            if exception:
                message = exception[0]
        context = record.get("fl_ctx", "")
        fields = dict(re.findall(r"(\w+)=([^,\]]*)", context)) if isinstance(context, str) else {}
        aborted_peer = fields.get("peer") if fields.get("peer_rc") == ReturnCode.TASK_ABORTED else None
        return (str(record.get("name", "unknown component")), message, location, application_frame, aborted_peer)
    return None


def failure_summary(result=None, *, since=None):
    """Summarize available job-local errors; never infer a global root cause.

    Supports simulator site logs and server/client logs included in downloaded
    POC/production results. Reads at most 20 files and the last 1 MiB of each.
    Repeated errors are grouped; traceback frames in application code are shown
    first. Missing/rotated/custom logs must not hide the original failure.
    ``since`` excludes untouched logs from an earlier simulator deployment.
    """
    lines = ["", "  Failure details"]
    try:
        groups = {}
        root = Path(result) if result else None
        candidates = set()
        client_logs_available = False
        if root:
            folders = [root, root / "workspace", *islice(root.glob("failure-logs-*"), _MAX_LOG_FILES)]
            for folder in folders:
                candidates.update((folder, *islice(folder.glob("*/"), _MAX_LOG_FILES)))
            files_read = 0
            for site_dir in sorted(candidates):
                selected = None
                for filename in ("log.json", "error_log.txt"):
                    path = site_dir / filename
                    if not path.is_file() or root.resolve() not in path.resolve().parents:
                        continue
                    if since is not None and path.stat().st_mtime < since:
                        continue
                    if files_read >= _MAX_LOG_FILES:
                        break
                    files_read += 1
                    if site_dir not in (root, root / "workspace") and site_dir.name != "server":
                        client_logs_available = True
                    try:
                        details = _first_error(path)
                    except (OSError, ValueError):
                        continue
                    if details and (selected is None or details[2]):
                        selected = (details, path)
                    if details and details[2]:
                        break  # A traceback is already available; avoid reading duplicate text.
                if selected:
                    details, path = selected
                    groups.setdefault(details, []).append(path)
        # An application traceback identifies the failing code more directly
        # than the resulting server abort. Do not order machines by wall clock.
        items = sorted(groups.items(), key=lambda item: not item[0][3])
        application_sites = {p.parent.name for details, paths in groups.items() if details[3] for p in paths}
        # Omit downstream abort records only when the named peer's application
        # traceback is available. An unrelated site's error is not sufficient.
        items = [item for item in items if item[0][4] not in application_sites]
        for index, ((component, message, location, _, _), paths) in enumerate(items[:3]):
            sites = ["server" if p.parent in (root, root / "workspace") else p.parent.name for p in paths]
            lines.extend(["", f"  {'Error' if index == 0 else 'Also reported':<8}  {_display(message)}"])
            lines.append(f"  Where     {_display(', '.join(sites), 160)} / {_display(component, 80)}")
            if location:
                lines.append(f"  Code      {_display(location, 160)}")
            logs = [p.with_name("error_log.txt") if p.with_name("error_log.txt").is_file() else p for p in paths]
            lines.append("  Logs      " + " · ".join(_display(p.relative_to(root), 160) for p in logs[:3]))
        if not items:
            lines.append("  No job error details are available locally.")
            lines.append("  Check server and client logs for the failed job.")
        else:
            lines.extend(["", "  Full tracebacks and additional messages are in the logs."])
            if not client_logs_available:
                lines.append("  Client logs are not included here; check them if the cause is unclear.")
    except Exception:
        # Formatting and artifact failures must never replace the job failure.
        lines.append("  Could not read error details. Check the job's server and client logs.")
    return wrap_log_message("\n".join(lines))
