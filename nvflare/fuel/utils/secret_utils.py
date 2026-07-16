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

"""Helpers for keeping secret values out of job definitions.

Job parameters (training script args, per-site config, extra config dicts) are written in clear
text into the generated job configuration files, which are exported, logged, and shared. They
must therefore never contain actual secret values. This module provides two complementary tools
to support that contract:

* Secret references: :func:`secret_ref` and :func:`secret_file_ref` build placeholders that can be
  used at supported runtime boundaries instead of actual secret values. Only placeholders are
  stored in generated job configuration; supported consumers call :func:`resolve_secret_refs`
  with values from the executing site's environment or mounted secret files at runtime.

* Secret detection: :func:`find_potential_secrets` and :func:`warn_on_potential_secrets` scan
  user-supplied parameter values for things that look like actual secrets (well-known token
  formats, password-like flags or keys, high-entropy strings) so mistakes can be flagged before
  the values end up in exported job configs and logs.

Messages produced by this module report only the flagged value's length, never any of its
characters, so warning text is safe to log and share.
"""

import math
import os
import re
import shlex
import warnings
from typing import Any, List, Mapping, NamedTuple, Optional

__all__ = [
    "PotentialSecretWarning",
    "UnsupportedSecretRefWarning",
    "SecretFinding",
    "SECRET_REF_PATTERN",
    "secret_ref",
    "secret_file_ref",
    "has_secret_refs",
    "resolve_secret_refs",
    "split_command_preserving_secret_refs",
    "find_potential_secrets",
    "warn_on_potential_secrets",
    "warn_on_unsupported_secret_refs",
    "warn_on_unsupported_secret_ref_keys",
    "warn_on_unsupported_secret_refs_outside_keys",
]

# Secret references are resolved on the *executing site* by supported runtime consumers.
# Environment references retain the concise ${secret:NAME} spelling; mounted-file references
# use ${secret:file:/path/to/key}. Whitespace and braces in file paths are intentionally
# unsupported so references remain portable across command and configuration encodings.
SECRET_REF_PATTERN = re.compile(r"\$\{secret:(?:file:(?P<file>[^{}\s\x00]+)|(?P<env>[A-Za-z_][A-Za-z0-9_]*))\}")

# Match both valid and malformed markers so tokenization cannot consume quotes or backslashes
# inside them before the resolver validates their syntax.
_SECRET_MARKER_PATTERN = re.compile(r"\$\{secret:[^}\r\n]*(?:\}|$)")

_ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_ASSIGNMENT_START_PATTERN = re.compile(
    r"""(?ix)(?=(?P<assignment>(?:^|[\s;=,("'`?&])(?:export\s+)?(?:\$env:)?"""
    r"""(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*))"""
)


class PotentialSecretWarning(UserWarning):
    """Warning category emitted when a job parameter looks like it contains an actual secret value."""


class UnsupportedSecretRefWarning(UserWarning):
    """Warning emitted when a secret reference appears outside a supported runtime boundary."""


class SecretFinding(NamedTuple):
    """One potential secret found by :func:`find_potential_secrets`.

    Attributes:
        location: where the value was found, e.g. ``"train_args (secret-named flag)"``.
        reason: why it was flagged, e.g. ``"GitHub token"``.
        preview: masked preview of the value, safe to log and share.
    """

    location: str
    reason: str
    preview: str


def secret_ref(env_var: str) -> str:
    """Build a secret reference placeholder for use in job parameters.

    The returned string carries no secret value and is safe to store in job configs. A supported
    runtime consumer on the executing site replaces it with the value of the named environment
    variable. See :mod:`nvflare.recipe.secrets` for supported parameter locations.

    Args:
        env_var: name of the environment variable that holds the secret on the executing site.

    Returns:
        The placeholder string ``${secret:<env_var>}``.

    Example:
        train_args=f"--api-key {secret_ref('MY_API_KEY')}"
    """
    if not isinstance(env_var, str) or not _ENV_VAR_NAME_PATTERN.match(env_var):
        raise ValueError("env_var must be a valid environment variable name ([A-Za-z_][A-Za-z0-9_]*)")
    return "${secret:" + env_var + "}"


def secret_file_ref(path: str) -> str:
    """Build a reference to a secret stored in a mounted text file on the executing site.

    This is intended for secrets projected as files, including Kubernetes Secret volume keys.
    The path itself is stored in the generated job config; the file content is read only on the
    executing site at runtime. One trailing newline commonly added by secret-file tooling is
    removed before injection.

    Args:
        path: runtime path of the mounted secret file. Absolute paths are strongly recommended;
            whitespace and brace characters are not supported.

    Returns:
        The placeholder string ``${secret:file:<path>}``.
    """
    if (
        not isinstance(path, str)
        or not path
        or not path.isprintable()
        or any(c in path for c in ("{", "}", "\0"))
        or any(c.isspace() for c in path)
    ):
        raise ValueError("path must be a non-empty printable secret file path without braces or whitespace")
    return "${secret:file:" + path + "}"


def has_secret_refs(value: Any) -> bool:
    """Return whether a string contains an environment or mounted-file secret reference."""
    return isinstance(value, str) and SECRET_REF_PATTERN.search(value) is not None


def resolve_secret_refs(value: Any, env: Optional[Mapping[str, str]] = None) -> Any:
    """Resolve secret references recursively without changing mapping keys.

    Intended to be called at runtime on the executing site, as late as possible and only on the
    in-memory value handed to user code. Environment references use ``${secret:NAME}``;
    mounted-file references use ``${secret:file:/path/to/key}``. Resolved values must never be
    logged or written back into config files.

    Args:
        value: string or nested mapping/list/tuple whose string values should be resolved.
        env: source of secret values; defaults to ``os.environ``.

    Returns:
        A resolved value. Containers are copied; mapping keys are preserved unchanged.

    Raises:
        ValueError: if reference syntax is invalid, a referenced environment variable is not set,
            or a referenced file cannot be read. Errors never include a resolved value, so they
            are safe to log.
    """
    source = os.environ if env is None else env

    def _sub(match: re.Match) -> str:
        env_name = match.group("env")
        if env_name is not None:
            if env_name not in source:
                raise ValueError(
                    f"cannot resolve secret reference '${{secret:{env_name}}}': "
                    f"environment variable '{env_name}' is not set on this site"
                )
            return source[env_name]

        path = match.group("file")
        if not path.isprintable():
            raise ValueError("cannot resolve secret reference: invalid secret reference syntax")
        try:
            with open(path, encoding="utf-8") as f:
                return f.read().removesuffix("\n")
        except (OSError, UnicodeError, ValueError):
            raise ValueError(
                f"cannot resolve secret file reference for path {path!r}: "
                f"secret file {path!r} cannot be read on this site"
            ) from None

    if isinstance(value, str):
        # Validate all markers before resolving any of them. This prevents malformed or manually
        # composed references from silently surviving as literals and avoids partial resolution
        # if another reference in the same value is invalid.
        if "${secret:" in SECRET_REF_PATTERN.sub("", value):
            raise ValueError("cannot resolve secret reference: invalid secret reference syntax")
        return SECRET_REF_PATTERN.sub(_sub, value)
    if isinstance(value, Mapping):
        return {key: resolve_secret_refs(item, env=source) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_secret_refs(item, env=source) for item in value]
    if isinstance(value, tuple):
        return tuple(resolve_secret_refs(item, env=source) for item in value)
    return value


def _shield_quoted_secret_ref_composites(command: str, marker_prefix: str):
    """Shield only quoted spans that contain a secret-ref marker.

    TaskScriptRunner historically uses whitespace-only splitting. Shielding these spans lets it
    support ``"Bearer ${secret:TOKEN}"`` without changing the treatment of unrelated quotes or
    backslashes elsewhere in the command.
    """
    quoted_prefix = f"{marker_prefix}QUOTED_"
    quoted_values = []
    parts = []
    index = 0

    def _is_quote_opener(position: int) -> bool:
        if position == 0:
            return True
        return command[position - 1].isspace() or command[position - 1] in "=([{,:"

    def _find_quote(start: int, quote: str) -> int:
        end = start
        while end < len(command):
            if command[end] == "\\" and end + 1 < len(command):
                end += 2
            elif command[end] == quote:
                break
            else:
                end += 1
        return end

    while index < len(command):
        quote = command[index]
        if quote not in {"'", '"'}:
            parts.append(quote)
            index += 1
            continue

        end = _find_quote(index + 1, quote)

        if end < len(command) and marker_prefix in command[index + 1 : end]:
            marker = f"{quoted_prefix}{len(quoted_values)}__"
            quoted_values.append((marker, command[index + 1 : end]))
            parts.append(marker)
            index = end + 1
        elif not _is_quote_opener(index):
            # Apostrophes and stray quotes embedded in an unquoted token are literal under the
            # runner's legacy whitespace split. Advancing one character lets a later real opener
            # use the same delimiter. The secret-bearing-span case above deliberately comes first
            # to retain support for shell-style concatenation such as ``prefix"Bearer ${secret:X}"``.
            parts.append(quote)
            index += 1
        elif end < len(command):
            # Keep matched non-secret-ref quoting byte-for-byte compatible with legacy splitting.
            # Skipping the entire pair also prevents its closing delimiter from being mistaken for
            # the opener of a later span.
            parts.append(command[index : end + 1])
            index = end + 1
        else:
            # Preserve an unmatched quote byte-for-byte, but continue from the next character.
            parts.append(quote)
            index += 1

    return "".join(parts), quoted_values


def split_command_preserving_secret_refs(command: str, posix: bool, group_secret_ref_quotes: bool = False) -> List[str]:
    """Tokenize a command without interpreting characters inside secret reference markers.

    Valid and malformed markers are temporarily replaced with inert tokens. Restoring them after
    tokenization ensures the runtime resolver sees exactly what the recipe supplied, including
    quote and backslash characters in mounted-file paths. ``posix=False`` retains the task script
    runner's legacy whitespace-only splitting; ``posix=True`` uses :func:`shlex.split`. With
    ``group_secret_ref_quotes=True``, only quoted spans containing references are grouped and
    unquoted; unrelated arguments retain the legacy behavior.
    """
    marker_prefix = "__NVFLARE_SECRET_MARKER_"
    while marker_prefix in command:
        marker_prefix = "_" + marker_prefix

    references = []

    def _shield(match: re.Match) -> str:
        marker = f"{marker_prefix}{len(references)}__"
        references.append((marker, match.group(0)))
        return marker

    shielded = _SECRET_MARKER_PATTERN.sub(_shield, command)
    quoted_values = []
    if group_secret_ref_quotes:
        shielded, quoted_values = _shield_quoted_secret_ref_composites(shielded, marker_prefix)
    tokens = shlex.split(shielded) if posix else shielded.split()
    for index, token in enumerate(tokens):
        for marker, quoted_value in quoted_values:
            token = token.replace(marker, quoted_value)
        for marker, reference in references:
            token = token.replace(marker, reference)
        tokens[index] = token
    return tokens


# Well-known credential formats. These are high-confidence: matching strings are almost
# certainly actual secrets, not identifiers or hyperparameters.
_KNOWN_TOKEN_PATTERNS = [
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("GitLab token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}")),
    ("API key (sk- format)", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("AWS access key ID", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{28,}")),
    ("JSON web token", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----")),
    ("credential embedded in URL", re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^/\s:@]+:[^/\s@]+@")),
]

# Flag/key names that suggest their value is a credential. Matches whole name segments
# ("api_key", "--auth-token") but not substrings inside larger words ("tokenizer", "max_tokens").
# A bare "key" segment is only matched as the final segment ("wandb_key" yes, "key_metric" no).
_SECRET_KEY_NAME_PATTERN = re.compile(
    r"(?i)(?:(?:^|[_\-.])(?:password|passwd|pwd|passphrase|secret|token|apikey|api[_\-]key|"
    r"accesskey|access[_\-]key|authtoken|auth[_\-]token|authorization|credentials?|privatekey|"
    r"private[_\-]key|clientsecret|client[_\-]secret|auth)(?=$|[_\-.])|[_\-.]key$)"
)

_BASE64_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=_-")
_HEX_CHARS = frozenset("0123456789abcdefABCDEF")

# Values that commonly appear after secret-named flags but are clearly not secrets.
_NON_SECRET_WORDS = frozenset({"true", "false", "none", "null", "yes", "no", "auto", "default", "disabled", "enabled"})

_LEGACY_REFERENCE_PATTERNS = (
    re.compile(r"^\$(?:[A-Z_][A-Z0-9_]*|\{[A-Z_][A-Z0-9_]*\})$"),
    re.compile(r"^%[A-Z_][A-Z0-9_]*%$"),
    re.compile(r"^\{[A-Z_][A-Z0-9_]*\}$"),
)


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts: dict = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(value)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _mask(value: str) -> str:
    """Return a masked value description containing no secret characters."""
    return f"'***' ({len(value)} chars)"


def _looks_like_path(value: str) -> bool:
    if value.startswith(("/", "./", "../", "~", "\\", "file:")):
        return True
    # Windows drive path, e.g. C:\... or C:/...
    return len(value) > 2 and value[1] == ":" and value[2] in ("/", "\\")


def _is_exempt_value(value: str) -> bool:
    """Values that are references or locations rather than secret material itself."""
    v = value.strip().strip("'\"")
    if len(v) < 6:
        return True
    if v.lower() in _NON_SECRET_WORDS:
        return True
    if "${secret:" in v:
        remainder = SECRET_REF_PATTERN.sub("", v)
        if "${secret:" in remainder:
            return False
        # A valid reference alone, or with a conventional authorization-scheme prefix, carries
        # no secret material. Other adjacent text is still scanned so a valid reference cannot
        # be used to hide a literal value in a secret-named field.
        remainder = remainder.strip(" \t\r\n'\"=:,;/-")
        if not remainder or remainder.lower() in {"bearer", "basic"}:
            return True
        return False
    # Require a complete, conventional placeholder. This prevents a literal such as
    # "$uperSecret123" from bypassing checks merely because it starts with '$'.
    if any(pattern.fullmatch(v) for pattern in _LEGACY_REFERENCE_PATTERNS):
        return True
    # Paths point at mounted secret files -- the recommended practice, not a leak.
    return _looks_like_path(v)


def _split_token_variants(text: str) -> List[List[str]]:
    """Return tokenizations needed to conservatively scan command-like text."""
    try:
        return [shlex.split(text, posix=True)]
    except ValueError:
        # Recipe parameters can contain incomplete shell quoting while they are being composed.
        # A plain whitespace split can separate the value of a secret-named flag from the rest of
        # its unterminated quoted span and make the detector miss it. Conversely, an unmatched
        # quote before a flag can make a lenient tokenizer absorb the flag into one token. Scan
        # both interpretations; findings are deduplicated before they are returned.
        lenient_tokens = _split_tokens_leniently(text)
        whitespace_tokens = text.split()
        if lenient_tokens == whitespace_tokens:
            return [lenient_tokens]
        return [lenient_tokens, whitespace_tokens]


def _split_tokens_leniently(text: str) -> List[str]:
    """Best-effort shell tokenization that keeps unterminated quoted spans together."""
    tokens: List[str] = []
    token: List[str] = []
    quote = None
    escaped = False

    for char in text:
        if escaped:
            token.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif quote:
            if char == quote:
                quote = None
            else:
                token.append(char)
        elif char in {"'", '"'}:
            quote = char
        elif char.isspace():
            if token:
                tokens.append("".join(token))
                token = []
        else:
            token.append(char)

    if escaped:
        token.append("\\")
    if token:
        tokens.append("".join(token))
    return tokens


def _iter_flag_findings(tokens: List[str], location: str, include_inline: bool = True):
    """Yield findings for command-line style '--secret-named-flag value' pairs."""
    for i, token in enumerate(tokens):
        if not token.startswith("-"):
            continue
        name = token.lstrip("-")
        value = None
        if "=" in name:
            if not include_inline:
                continue
            name, value = name.split("=", 1)
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            value = tokens[i + 1]
        if not name or value is None:
            continue
        if _SECRET_KEY_NAME_PATTERN.search(name) and not _is_exempt_value(value):
            yield SecretFinding(
                location=f"{location} (secret-named flag)",
                reason="value of a secret-named flag",
                preview=_mask(value.strip().strip("'\"")),
            )


def _extract_assignment_value(text: str, start: int, unquoted_limit: int) -> tuple[str, int]:
    if start >= len(text):
        return "", start

    quote = text[start]
    if quote in {"'", '"'}:
        end = start + 1
        while end < len(text):
            if text[end] == "\\" and end + 1 < len(text):
                end += 2
            elif text[end] == quote:
                return text[start : end + 1], end
            else:
                end += 1
        return text[start:], len(text)

    limit = min(unquoted_limit, len(text))
    end = start
    while end < limit and not text[end].isspace() and text[end] != ";":
        end += 1
    return text[start:end], start


def _iter_assignment_findings(text: str, location: str):
    """Yield findings for shell, option-wrapped, and PowerShell secret assignments."""
    matches = list(_ASSIGNMENT_START_PATTERN.finditer(text))
    index = 0
    while index < len(matches):
        match = matches[index]
        name = match.group("name")
        value_start = match.end("assignment")
        unquoted_limit = len(text)
        if index + 1 < len(matches):
            next_start = matches[index + 1].start("assignment")
            # Optional prefixes (for example an opening quote plus ``export``) can produce
            # duplicate candidates whose start precedes this same assignment's value.
            if next_start > value_start:
                unquoted_limit = next_start
        value, skip_nested_before = _extract_assignment_value(text, value_start, unquoted_limit)
        if _SECRET_KEY_NAME_PATTERN.search(name) and not _is_exempt_value(value):
            yield SecretFinding(
                location=f"{location} (secret-named assignment)",
                reason="value of a secret-named assignment",
                preview=_mask(value.strip().strip("'\"")),
            )
        index += 1
        while index < len(matches) and matches[index].start("assignment") < skip_nested_before:
            index += 1


def _entropy_finding(token: str, location: str) -> Optional[SecretFinding]:
    value = token.strip().strip("'\",;")
    if len(value) < 20 or _is_exempt_value(value):
        return None
    if all(c in _HEX_CHARS for c in value):
        if len(value) >= 32 and _shannon_entropy(value) > 3.0:
            return SecretFinding(location, "high-entropy hex string", _mask(value))
        return None
    if all(c in _BASE64_CHARS for c in value) and _shannon_entropy(value) > 4.5:
        return SecretFinding(location, "high-entropy string", _mask(value))
    return None


def _scan_string(text: str, location: str, findings: List[SecretFinding]) -> None:
    if "${secret:" in SECRET_REF_PATTERN.sub("", text):
        findings.append(SecretFinding(location, "malformed secret reference", _mask(text)))
    for reason, pattern in _KNOWN_TOKEN_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(SecretFinding(location, reason, _mask(match.group(0))))
    findings.extend(_iter_assignment_findings(text, location))
    for tokens in _split_token_variants(text):
        findings.extend(_iter_flag_findings(tokens, location))
        for token in tokens:
            finding = _entropy_finding(token, location)
            if finding:
                findings.append(finding)


def _scan_value(value: Any, location: str, findings: List[SecretFinding]) -> None:
    if isinstance(value, str):
        _scan_string(value, location, findings)
    elif isinstance(value, Mapping):
        for index, (key, item) in enumerate(value.items()):
            key_findings: List[SecretFinding] = []
            if isinstance(key, str):
                _scan_string(key, f"{location} (mapping key #{index})", key_findings)
            findings.extend(key_findings)
            # Never copy mapping keys into warning locations: a heuristic miss in a key must not
            # leak that key when a descendant value produces a finding.
            child_location = f"{location} (mapping value #{index})"
            if isinstance(key, str) and isinstance(item, str):
                if _SECRET_KEY_NAME_PATTERN.search(key) and not _is_exempt_value(item):
                    findings.append(
                        SecretFinding(
                            location=child_location,
                            reason="value of a secret-named key",
                            preview=_mask(item.strip().strip("'\"")),
                        )
                    )
            _scan_value(item, child_location, findings)
    elif isinstance(value, (list, tuple)):
        if value and all(isinstance(item, str) for item in value):
            # Preserve argv adjacency: recursively scanning each item cannot associate a
            # secret-named flag with its value in the following element. Inline assignments
            # are still found by the per-item scan below.
            findings.extend(_iter_flag_findings(list(value), location, include_inline=False))
        for index, item in enumerate(value):
            _scan_value(item, f"{location}[{index}]", findings)


def find_potential_secrets(value: Any, location: str = "value") -> List[SecretFinding]:
    """Scan a parameter value for anything that looks like an actual secret.

    Strings are checked for well-known credential formats, secret-named command-line flags with
    inline values, credentials embedded in URLs, and high-entropy tokens. Dicts and lists are
    scanned recursively; dict entries whose key names suggest a credential (password, token,
    api_key, ...) are flagged when their value looks like actual secret material. Stand-alone,
    valid secret references (``${secret:NAME}``), env-var/config placeholders, and
    file paths are not flagged -- those are the recommended ways to hand secrets to a job.

    This is a heuristic intended to catch mistakes, not a guarantee: absence of findings does not
    prove a value is safe.

    Args:
        value: the parameter value to scan (str, dict, list, or any nesting of them).
        location: human-readable description of where the value came from; used in findings.

    Returns:
        List of :class:`SecretFinding`, deduplicated by location and preview. Findings contain
        only masked previews and are safe to log.
    """
    findings: List[SecretFinding] = []
    _scan_value(value, location, findings)
    deduped: List[SecretFinding] = []
    seen = set()
    for finding in findings:
        key = (finding.location, finding.preview)
        if key not in seen:
            seen.add(key)
            deduped.append(finding)
    return deduped


def _emit_safe_warning(message: str, category: type[Warning]) -> None:
    """Emit a warning without exposing a caller source line that may contain a secret.

    Warning-as-error policies are deliberately neutralized for these diagnostics. Letting the
    warning exception escape would render caller frames in a traceback, including an inline recipe
    parameter containing the very secret this scanner is trying to keep out of logs.
    """
    try:
        warnings.warn_explicit(message, category, filename="<nvflare-secret-scan>", lineno=1)
    except category:
        with warnings.catch_warnings():
            warnings.simplefilter("always", category)
            warnings.warn_explicit(message, category, filename="<nvflare-secret-scan>", lineno=1)


def warn_on_potential_secrets(value: Any, context: str) -> List[SecretFinding]:
    """Scan a parameter value and emit a :class:`PotentialSecretWarning` for each finding.

    Warning messages contain only masked previews, never the flagged values, so they are safe to
    appear in logs that get shared.

    Args:
        value: the parameter value to scan.
        context: where the value came from, e.g. ``"recipe parameter 'train_args'"``.

    Returns:
        The list of findings (empty if nothing was flagged).
    """
    findings = find_potential_secrets(value, location=context)
    for finding in findings:
        _emit_safe_warning(
            f"Potential secret value in {finding.location}: {finding.reason} {finding.preview}. "
            "Job parameters are written in clear text into the generated job configuration and may "
            "appear in logs and exported job folders - never put actual secret values in them. "
            "Read secrets from site environment variables or mounted secret files inside your "
            "training code, or pass a placeholder created with nvflare.recipe.secrets.secret_ref() "
            "or secret_file_ref() in a supported runtime parameter; see nvflare.recipe.secrets.",
            PotentialSecretWarning,
        )
    return findings


def _contains_secret_ref(value: Any) -> bool:
    if isinstance(value, str):
        return has_secret_refs(value)
    if isinstance(value, Mapping):
        return any(_contains_secret_ref(key) or _contains_secret_ref(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_secret_ref(item) for item in value)
    return False


def warn_on_unsupported_secret_refs(value: Any, context: str) -> bool:
    """Warn when a value contains references that its runtime consumer will not resolve."""
    if not _contains_secret_ref(value):
        return False
    _emit_safe_warning(
        f"Secret reference in {context} is outside a supported runtime boundary and will remain "
        "a literal placeholder. Read the secret from the site environment or mounted file inside "
        "user code; see nvflare.recipe.secrets for supported reference locations.",
        UnsupportedSecretRefWarning,
    )
    return True


def _contains_secret_ref_in_keys(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_secret_ref(key) or _contains_secret_ref_in_keys(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_secret_ref_in_keys(item) for item in value)
    return False


def warn_on_unsupported_secret_ref_keys(value: Any, context: str) -> bool:
    """Warn when nested mapping keys contain references, since keys are never resolved."""
    if not _contains_secret_ref_in_keys(value):
        return False
    _emit_safe_warning(
        f"Secret reference in a mapping key under {context} will remain a literal placeholder. "
        "Dictionary keys are never resolved; use a fixed key and put the reference in its value.",
        UnsupportedSecretRefWarning,
    )
    return True


def _contains_secret_ref_outside_keys(
    value: Any, supported_value_keys: frozenset[str], supported_value_depth: int
) -> bool:
    if isinstance(value, str):
        return has_secret_refs(value)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _contains_secret_ref(key):
                return True
            if (
                supported_value_depth == 1
                and isinstance(key, str)
                and key in supported_value_keys
                and isinstance(item, str)
            ):
                # These runtime boundaries consume direct string fields. Do not exempt a nested
                # mapping/list merely because its parent happens to use a supported field name.
                continue
            if _contains_secret_ref_outside_keys(item, supported_value_keys, max(supported_value_depth - 1, 0)):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(
            _contains_secret_ref_outside_keys(item, supported_value_keys, supported_value_depth) for item in value
        )
    return False


def warn_on_unsupported_secret_refs_outside_keys(
    value: Any,
    supported_value_keys: set[str],
    context: str,
    supported_value_depth: int = 1,
) -> bool:
    """Warn when references occur outside explicitly supported mapping values.

    ``supported_value_depth`` controls the mapping depth at which supported keys are exempted.
    Per-site configuration uses depth 2 for its site key and immediate field values.
    """
    if not _contains_secret_ref_outside_keys(value, frozenset(supported_value_keys), supported_value_depth):
        return False
    supported = ", ".join(sorted(supported_value_keys))
    _emit_safe_warning(
        f"Secret reference in {context} is outside a supported field and will remain a literal "
        f"placeholder. Only these fields resolve references: {supported}.",
        UnsupportedSecretRefWarning,
    )
    return True
