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

"""Deterministic v1 admission lints for NVFLARE-owned agent skills.

DESIGN INVARIANT -- lint engine independence (do not revert):
This engine reads only the ``skills/`` tree, including each skill's repo-local
``evals/`` directory. It must NOT read ``docs/design/*.md`` or rely on
offline-only catalog metadata. ``SKILL.md`` is a runtime artifact loaded by the
agent; fields validated here must be runtime or public skill metadata, not
private lint scratch data. Eval suites are evaluation metadata, distinct from
the forbidden ``docs_root`` and from runtime guidance scanned by this lint.

Concretely:
- Group skills for ``skill-trigger-overlap-lint`` by deterministic skill-name
  families (see ``_trigger_overlap_group``), not by frontmatter ``category``
  values or by a product-catalog table parsed from design docs.
- ``category`` is valid public SKILL.md metadata for publishable skills; it is
  not a trigger-overlap grouping source or a docs-catalog sync key for this
  engine.
- Do not add a ``docs_root`` parameter or a ``--docs-root`` flag back to this
  engine, and do not re-introduce ``skill-catalog-category-lint`` /
  ``agent-doc-crosslink-lint`` here.
- Catalog/publication sync (skill listed in the human product catalog) is a
  docs concern: put it in a SEPARATE docs check, not in this skill engine.

Rationale and history: docs/design/skills_architecture.md "Lint engine
independence". A prior change coupled this engine to design docs and made
catalog synchronization part of the skill lint runner; that coupling was
reverted on purpose. Keep category validation local to SKILL.md frontmatter, and
keep docs/catalog synchronization in separate docs tooling.
"""

import json
import os
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from .frontmatter import (
        PUBLIC_EXEMPT_STATUS,
        SKILL_FILE_NAME,
        SKILL_NAME_RE,
        RegularFileTooLargeError,
        SkillValidationResult,
        parse_skill_frontmatter,
        read_regular_text_file,
        should_skip_skill_dir,
        skill_metadata,
        validate_skill_dir,
    )
except ImportError as e:
    # Only fall back to the bare-script import path for the script-vs-package
    # case (e.name is None: "attempted relative import with no known parent
    # package"). A genuinely missing third-party dep (e.g. PyYAML) has a real
    # e.name and must surface with its true message instead of being masked.
    if e.name is not None:
        raise
    from frontmatter import (
        PUBLIC_EXEMPT_STATUS,
        SKILL_FILE_NAME,
        SKILL_NAME_RE,
        RegularFileTooLargeError,
        SkillValidationResult,
        parse_skill_frontmatter,
        read_regular_text_file,
        should_skip_skill_dir,
        skill_metadata,
        validate_skill_dir,
    )

LINT_SKILL_FRONTMATTER = "skill-frontmatter-lint"
LINT_SKILL_MD_SIZE = "skill-md-size-lint"
LINT_SKILL_TRIGGER = "skill-trigger-lint"
LINT_SKILL_TRIGGER_OVERLAP = "skill-trigger-overlap-lint"
LINT_SKILL_GLOBAL_NEGATIVE = "skill-global-negative-lint"
LINT_SKILL_POLICY_COVERAGE = "skill-policy-coverage-lint"
LINT_SKILL_PROCESS_METRIC = "skill-process-metric-lint"
LINT_SKILL_COMMAND_DRIFT = "skill-command-drift-lint"
LINT_SKILL_HELPER_SCRIPT = "skill-helper-script-lint"
LINT_SKILL_FIXTURE = "skill-fixture-lint"
LINT_SKILL_RUNTIME_BOUNDARY = "skill-runtime-boundary-lint"
LINT_SKILL_DEPENDENCY_INSTALL_SAFETY = "skill-dependency-install-safety-lint"

FINDING_ERROR = "error"
FINDING_WARNING = "warning"
FINDING_INFO = "info"
SKILL_MD_MAX_LINES = 200
SKILL_MD_ADVISORY_WORDS = 2000
MAX_SKILL_TEXT_FILE_BYTES = 512 * 1024
DEFAULT_MAX_TRIGGER_OVERLAP_SKILLS = 200

_SIZE_EXCEPTION_MARKERS = (
    "nvflare-lint: allow skill-md-size-lint",
    "skill-md-size-lint: approved-exception",
)
_TRIGGER_TERMS = (
    "trigger",
    "use when",
    "when to use",
    "use this skill",
    "do not use",
    "do-not-use",
    "boundary",
)
_BOUNDARY_TERMS = ("do not use", "do-not-use", "use boundary", "boundary", "negative")
_NORMATIVE_RE = re.compile(r"\b(must|must not|required|prohibited|approval)\b", re.IGNORECASE)
_BACKTICK_NVFLARE_RE = re.compile(r"`(nvflare(?:\s+[^`]+)?)`")
_SAFE_COMMAND_TOKEN_RE = re.compile(r"^(?:--?[A-Za-z0-9][\w-]*(?:=[^\s`;&|]+)?|[A-Za-z0-9_./:=+@%,-]+|<[^>\n]+>)$")
_SIGNIFICANT_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{2,}")
_STOPWORDS = {
    "and",
    "for",
    "the",
    "this",
    "that",
    "with",
    "into",
    "from",
    "using",
    "when",
    "skill",
    "nvflare",
    "flare",
    "federated",
    "workflow",
}
_KNOWN_NVFLARE_ROOT_COMMANDS = {
    "agent",
    "authz-preview",
    "cert",
    "config",
    "dashboard",
    "deploy",
    "job",
    "package",
    "poc",
    "preflight-check",
    "provision",
    "recipe",
    "simulator",
    "study",
    "system",
}
_KNOWN_AGENT_COMMANDS = {"info", "inspect"}
_KNOWN_AGENT_INSPECT_CAPABILITIES = {"data", "source"}
_KNOWN_AGENT_FLAGS = {
    "agent": {"--format", "--schema"},
    "agent info": {"--format", "--schema"},
    "agent inspect": {"--format", "--schema"},
    "agent inspect data": {"--format", "--max-file-bytes", "--max-files", "--redact", "--schema"},
    "agent inspect source": {"--format", "--max-file-bytes", "--max-files", "--redact", "--schema"},
}

_DEPENDENCY_INSTALL_TERMS_RE = re.compile(
    r"\b(?:dependenc\w*|install\w*|package\w*|requirements?|(?:pip|uv|poetry|pipenv)\s+add)\b", re.IGNORECASE
)
_DEPENDENCY_ACTION_PATTERN = (
    r"(?:install\w*|download\w*|us(?:e|es|ed|ing)|execut\w*|fetch\w*|sync\w*|add(?:s|ed|ing)?|"
    r"upgrad\w*|resolv\w*|appl(?:y|ies|ied|ying))"
)
_DEPENDENCY_ACTION_RE = re.compile(rf"\b{_DEPENDENCY_ACTION_PATTERN}\b", re.IGNORECASE)
_DEPENDENCY_NOUN_PATTERN = r"(?:dependenc\w*|packages?|requirements?)"
# Read-only verbs change nothing, so they need no install confirmation. Matching a
# verb is not enough: the exemption is granted only to a whole canonical sentence
# shape (below), so an unrecognized mutating verb anywhere in the clause withdraws
# it. Recognizing shapes rather than growing a verb list is what keeps this
# fail-closed -- the costly error for a safety lint is the miss, not the false
# positive. ``review`` is safe here because this vocabulary is consulted only by
# the confirmation check; the separate "without reviewing sources" matcher is
# unaffected.
# Inflections are spelled out rather than suffixed with ``\w*``: a greedy stem
# swallows a different verb whose prefix happens to match, so "checkout packages"
# is not read as an inflection of "check".
_READ_ONLY_DEPENDENCY_VERB_PATTERN = (
    r"(?:inspects?|inspect(?:ed|ing|ion)|reads?|reading|lists?|list(?:ed|ing)"
    r"|views?|view(?:ed|ing)|shows?|show(?:ed|ing)|examine[sd]?|examin(?:ing|ation)"
    r"|audits?|audit(?:ed|ing)|reviews?|review(?:ed|ing)|quer(?:y|ies|ied|ying)"
    r"|enumerate[sd]?|enumerating|prints?|print(?:ed|ing)|displays?|display(?:ed|ing)"
    r"|checks?|check(?:ed|ing))"
)
# A preposition followed by modifiers and a gerund can attach a second action to
# an otherwise read-only phrase: "dependencies must be inspected by only
# fetching packages". Scan the tail's words once rather than rescanning the
# remaining tail for every preposition or using an exponentially backtracking
# repeated regex group.
_ACTION_INTRODUCING_PREPOSITIONS = frozenset({"by", "via", "through", "with", "for", "during", "upon", "when"})
_POLICY_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'’-]*")
# Words that end in ``ing`` but can be nouns only when followed by a recognized
# agent noun. Keep this mapping deliberately narrow: an unknown gerund such as
# "fetching" must remain action-introducing even in "by a fetching script",
# and "engineering packages" must not inherit the "engineering team" exemption.
_NON_ACTION_GERUND_AGENT_NOUNS = {"engineering": frozenset({"team"})}
# An object word may not be a coordinator (which could attach a second action) or
# a recognized mutating verb, and may not be ``to``, which introduces an
# infinitive: "inspect package metadata to add packages".
_READ_ONLY_OBJECT_WORD_PATTERN = (
    r"(?!(?:and|or|nor|then|plus|also|while|before|after|but|to)\b)"
    rf"(?!{_DEPENDENCY_ACTION_PATTERN}\b)[A-Za-z0-9_.'’-]+"
)
# "check out" acquires a dependency where the bare "check" only reads one. The
# particle is disqualifying for ``check`` alone -- "print out" and "list out"
# stay read-only -- and in either position: "check out packages", "checking
# packages out". ``out`` must be a standalone token so "check out-of-date
# packages" remains a read.
_CHECK_OUT_ACQUISITION_RE = re.compile(r"\s*check(?:s|ed|ing)?\b(?:\s+\S+)*?\s+out(?:\s|$)", re.IGNORECASE)
_READ_ONLY_ACTION_PHRASE_RE = re.compile(
    rf"(?P<verb>{_READ_ONLY_DEPENDENCY_VERB_PATTERN})\b(?:\s+{_READ_ONLY_OBJECT_WORD_PATTERN})*",
    re.IGNORECASE,
)
# The same read done in the passive voice: "package metadata must be inspected".
# Reuse the active phrase's object-word rules on both sides of the passive verb
# so the match must consume the whole clause and cannot absorb a coordinator,
# infinitive, or recognized mutating action.
_READ_ONLY_PASSIVE_PHRASE_RE = re.compile(
    rf"(?:{_READ_ONLY_OBJECT_WORD_PATTERN}\s+)+"
    r"(?:(?:must|should|shall|can|could|may|might|will|would)\s+(?:be|have\s+been)\s+"
    r"|(?:is|are|was|were)\s+(?:being\s+)?|(?:has|have|had)\s+been\s+|to\s+be\s+)"
    r"(?P<verb>inspected|read|listed|viewed|shown|examined|audited|reviewed|queried"
    r"|enumerated|printed|displayed|checked)\b"
    rf"(?:\s+{_READ_ONLY_OBJECT_WORD_PATTERN})*",
    re.IGNORECASE,
)
# A coordinated series shares one negation: in "never download, install, or
# execute dependencies", the later verbs are still governed by "never". Only
# connectives, dependency nouns, and further actions may appear between them.
_DEPENDENCY_ACTION_SERIES_GAP_RE = re.compile(
    rf"(?:[\s,;/]|\b(?:and|or|nor|then)\b|\b{_DEPENDENCY_ACTION_PATTERN}\b|\b{_DEPENDENCY_NOUN_PATTERN}\b)*",
    re.IGNORECASE,
)
# A coordinator can attach an action whose verb is outside the recognized
# vocabulary, so its presence withdraws any "nothing else acts here" assumption.
_CLAUSE_COORDINATOR_RE = re.compile(r"[,;/]|\b(?:and|or|nor|then|plus|also|while|before|after)\b", re.IGNORECASE)
# A clause holding nothing but a negation before "without X" is verb ellipsis:
# the repeated action verb is dropped, as in "but never without user approval".
_ELLIPTICAL_NEGATION_RE = re.compile(
    r"\s*(?:never|not|do(?:es)?\s+not|do(?:es)?n['’]t|must\s+not|should\s+not|shall\s+not|may\s+not|"
    r"cannot|can\s+not|won['’]t|will\s+not)"
    r"(?:\s+(?:ever|preemptively|automatically|directly))?\s*",
    re.IGNORECASE,
)
_DEPENDENCY_CONFIRMATION_BYPASS_RES = (
    re.compile(
        r"\b(?:dependenc\w*|install\w*|package\w*|requirements?)\b[^.!?;]{0,160}"
        r"\bnever\s+(?:be\s+)?(?:preceded|followed)\s+by\b[^.!?;]{0,120}"
        r"\b(?:prompt|approval|confirmation)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:dependenc\w*|install\w*|package\w*|requirements?)\b[^.!?;]{0,120}"
        r"\b(?:do\s+not|don't|never)\s+(?:preemptively\s+)?(?:ask|prompt)\b[^.!?;]{0,100}"
        r"\b(?:approval|confirmation)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:dependenc\w*|install\w*|package\w*|requirements?)\b[^.!?;]{0,120}"
        r"\b(?:requires?|needs?)\s+no\s+(?:user\s+)?(?:approval|confirmation|consent)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:user\s+)?(?:approval|confirmation|consent)\s+(?:is\s+)?(?:required|needed)\b"
        r"[^.!?;]{0,120}\b(?:dependenc\w*|install\w*|package\w*|requirements?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:dependenc\w*|install\w*|package\w*|requirements?)\b[^.!?;]{0,180}"
        r"\bnever\s+emit\b[^.!?;]{0,80}\b(?:approval|confirmation)\s+prompts?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:skip\s+(?:the\s+)?(?:confirmation|approval)\s+prompt|silently\s+install)\b"
        r"[^.!?;]{0,140}\b(?:dependenc\w*|package\w*|requirements?)\b|"
        r"\b(?:dependenc\w*|package\w*|requirements?)\b[^.!?;]{0,140}"
        r"\b(?:skip\s+(?:the\s+)?(?:confirmation|approval)\s+prompt|silently\s+install)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+need\s+to\s+(?:ask|prompt)\b[^.!?;]{0,120}"
        r"\b(?:dependenc\w*|install\w*|package\w*|requirements?)\b|"
        r"\b(?:dependenc\w*|install\w*|package\w*|requirements?)\b[^.!?;]{0,120}"
        r"\bno\s+need\s+to\s+(?:ask|prompt)\b",
        re.IGNORECASE,
    ),
)
_DEPENDENCY_CONFIRMATION_WITHOUT_RE = re.compile(
    r"(?P<without_clause>\bwithout\s+(?:(?:asking|prompting)\s+(?:the\s+)?user\s+(?:for\s+)?|"
    r"(?:explicit|prior)\s+|(?:the\s+)?user['’]s\s+|user\s+)?(?:approval|confirmation|consent|permission)\b)",
    re.IGNORECASE,
)
_WITHOUT_CLAUSE_PROHIBITION_TAIL_RE = re.compile(
    r"\s*(?:(?:is\s+)?(?:strictly\s+)?(?:prohibited|forbidden|disallowed|banned)"
    r"|is\s+not\s+(?:allowed|permitted)"
    r"|(?:is\s+)?never\s+(?:allowed|permitted))\b",
    re.IGNORECASE,
)
_WITHOUT_CLAUSE_BOUNDARY_RE = re.compile(
    r",\s*(?:(?:and\s+)?then|but|yet|however|although|though|subsequently|afterwards?|next|"
    r"(?P<coordinator>and|or))\b"
    r"|\s+(?:but|however|although|though)\s+|\|",
    re.IGNORECASE,
)
_DEPENDENCY_CONFIRMATION_REQUEST_SUPPRESSION_RE = re.compile(
    r"\b(?:do(?:es)?\s+not|do(?:es)?n['’]t|must\s+not|shall\s+not|should\s+never|never)\s+"
    r"(?:preemptively\s+)?(?:ask|prompt)\b[^.!?;]{0,120}"
    r"\b(?:whether\s+to|before|prior\s+to|for\s+(?:an?\s+)?(?:approval|confirmation|permission)"
    r"(?:\s+(?:before|prior\s+to|when))?)\b[^.!?;]{0,100}"
    r"\b(?:dependenc\w*|install\w*|package\w*|requirements?|(?:pip|uv|poetry|pipenv)\s+add)\b",
    re.IGNORECASE,
)
_DEPENDENCY_AUDIT_FIRST_RE = re.compile(
    r"\b(?:do\s+not|don't|never)\s+(?:preemptively\s+)?(?:ask|prompt)\b[^.!?;]{0,180}"
    r"\b(?:before|prior\s+to)\b[^.!?;]{0,120}"
    r"\b(?:audit\w*|review\w*|vet\w*|classif\w*|flag\w*)\b",
    re.IGNORECASE,
)
_DEPENDENCY_POST_AUDIT_CONFIRMATION_RES = (
    re.compile(
        r"\b(?:after|following|once)\b[^.!?;]{0,80}"
        r"\b(?:audit\w*|review\w*|vet\w*|classif\w*)\b[^.!?;]{0,160}"
        r"\b(?:obtain|request|receive|require|wait\s+for)\b[^.!?;]{0,80}"
        r"\b(?:explicit\s+)?(?:user\s+)?(?:approval|confirmation|consent)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:obtain|request|receive|require|wait\s+for)\b[^.!?;]{0,80}"
        r"\b(?:explicit\s+)?(?:user\s+)?(?:approval|confirmation|consent)\b[^.!?;]{0,120}"
        r"\b(?:after|following|once)\b[^.!?;]{0,80}"
        r"\b(?:audit\w*|review\w*|vet\w*|classif\w*)\b",
        re.IGNORECASE,
    ),
)
_DEPENDENCY_REVIEW_WITHOUT_RE = re.compile(
    r"(?P<without_clause>\bwithout\s+(?:(?:an?|any|the)\s+)?(?:audit\w*|review\w*|"
    r"vet(?:s|ted|ting)?|classif\w*|flag\w*|check(?:ing|ed|s)?(?:\s+(?:their|the|package))?\s+sources?)\b)",
    re.IGNORECASE,
)
_DEPENDENCY_REVIEW_BYPASS_RES = (
    re.compile(
        r"\bwithout\s+asking\b[^.!?;]{0,80}\b(?:audit\w*|review\w*|vet\w*|classif\w*|flag\w*)\b"
        r"[^.!?;]{0,100}\b(?:dependenc\w*|install\w*|package\w*|requirements?|sources?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:does\s+not|do\s+not|don't|never)\s+"
        r"(?:audit\w*|review\w*|vet\w*|classif\w*|flag\w*)"
        r"(?:\s*,\s*(?:audit\w*|review\w*|vet\w*|classif\w*|flag\w*))*"
        r"(?:\s*,?\s*(?:or|and)\s+(?:audit\w*|review\w*|vet\w*|classif\w*|flag\w*))?"
        r"\s+(?:(?:the|any)\s+)?(?:dependenc\w*|install\w*|package\w*|requirements?|sources?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:skip|bypass|omit)\w*\b[^.!?;]{0,80}"
        r"\b(?:audit\w*|review\w*|vet(?:s|ted|ting)?|classif\w*|flag\w*)\b[^.!?;]{0,120}"
        r"\b(?:dependenc\w*|install\w*|package\w*|requirements?|sources?)\b",
        re.IGNORECASE,
    ),
)
_BARE_CONFIRMATION_BYPASS_RE = re.compile(
    r"^(?:(?:do(?:es)?\s+not|do(?:es)?n['’]t|must\s+not|shall\s+not|should\s+never|never)\s+"
    r"(?:preemptively\s+)?(?:ask|prompt)\b"
    r"[^.!?;]{0,80}\b(?:approval|confirmation|consent|permission)\b|"
    r"(?:skip\s+(?:the\s+)?(?:confirmation|approval)\s+prompt|silently\s+install|"
    r"there\s+is\s+no\s+need\s+to\s+(?:ask|prompt)|never\s+emit[^.!?;]{0,60}"
    r"(?:approval|confirmation)\s+prompts?)[^.!?;]*)[.!?;]?$",
    re.IGNORECASE,
)
_BARE_CONFIRMATION_DENIAL_RE = re.compile(
    r"^(?:without\s+(?:(?:explicit|prior)\s+|(?:the\s+)?user['’]s\s+|user\s+)?"
    r"(?:approval|confirmation|consent|permission)|"
    r"(?:requires?|needs?)\s+no\s+(?:further\s+)?(?:user\s+)?(?:approval|confirmation|consent|permission)|"
    r"no\s+(?:user\s+)?(?:approval|confirmation|consent|permission)\s+(?:is\s+)?(?:required|needed))[.!?;]?$",
    re.IGNORECASE,
)
_BARE_CONFIRMATION_BYPASS_CLAUSE_RE = re.compile(
    r"(?:^|,\s*)(?:do\s+not|don't|never)\s+(?:preemptively\s+)?(?:ask|prompt)\b[^.!?;,]{0,80}"
    r"\b(?:approval|confirmation|consent)\b\s*(?=,|$)",
    re.IGNORECASE,
)
_NEGATED_DEPENDENCY_ACTION_RE = re.compile(
    r"\b(?:never|do(?:es)?\s+not|do(?:es)?n['’]t|must\s+not|should\s+not|shall\s+not|may\s+not|"
    r"cannot|can\s+not|won['’]t|will\s+not)\b"
    r"\s+(?:(?:preemptively|ever|automatically|directly)\s+)?"
    rf"(?P<action>{_DEPENDENCY_ACTION_PATTERN})\b[^.!?;]{{0,100}}?"
    rf"\b{_DEPENDENCY_NOUN_PATTERN}\b",
    re.IGNORECASE,
)
_PASSIVE_NEGATED_DEPENDENCY_ACTION_RE = re.compile(
    rf"\b{_DEPENDENCY_NOUN_PATTERN}\b[^.!?;]{{0,80}}"
    r"\b(?:(?:must|should|shall|may|can|could|will|would)\s+not\s+(?:be\s+)?|"
    r"(?:cannot|can['’]t)\s+(?:be\s+)?|"
    r"(?:is|are|was|were|has|have|had)\s+not\s+(?:been\s+|being\s+)?)"
    rf"(?P<action>{_DEPENDENCY_ACTION_PATTERN})\b",
    re.IGNORECASE,
)
_PROHIBITED_DEPENDENCY_ACTION_RE = re.compile(
    rf"\b(?:dependenc\w*|packages?|requirements?)[^.!?;]{{0,80}}\b{_DEPENDENCY_ACTION_PATTERN}\b[^.!?;]{{0,80}}"
    r"\b(?:prohibited|forbidden|disallowed|banned)\b"
    rf"|\b{_DEPENDENCY_ACTION_PATTERN}\b[^.!?;]{{0,80}}\b(?:dependenc\w*|packages?|requirements?)\b"
    r"[^.!?;]{0,80}"
    r"\b(?:prohibited|forbidden|disallowed|banned)\b",
    re.IGNORECASE,
)
_PROHIBITED_DEPENDENCY_CONTEXT_RE = re.compile(
    rf"\b{_DEPENDENCY_NOUN_PATTERN}\b[^.!?;]{{0,100}}\b(?:is|are|was|were)\s+"
    r"(?:strictly\s+)?(?:prohibited|forbidden|disallowed|banned)\b",
    re.IGNORECASE,
)
_DEPENDENCY_ACTION_AT_END_RE = re.compile(rf"\b{_DEPENDENCY_ACTION_PATTERN}\s*$", re.IGNORECASE)
_DEPENDENCY_ACTION_AT_START_RE = re.compile(rf"^\s*{_DEPENDENCY_ACTION_PATTERN}\b", re.IGNORECASE)
_MARKDOWN_ATX_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}(?:\s+|$)")
_MARKDOWN_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}(?:>\s*)+")
_MARKDOWN_BLOCKQUOTE_CONTINUATION_END_RE = re.compile(
    r"\b(?:a|an|and|are|as|at|be|been|being|before|by|can|could|did|do|does|for|from|if|in|into|is|may|might|"
    r"must|never|not|of|on|or|preceded|followed|shall|should|that|the|to|was|were|will|with|without|would)"
    r"(?:[:,])?(?:[`*_]+)?\s*$",
    re.IGNORECASE,
)
_MARKDOWN_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_MARKDOWN_LIST_ITEM_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_MARKDOWN_SENTENCE_END_RE = re.compile(r"[.!?;](?:[`*_]+)?\s*$")
_MARKDOWN_STRUCTURAL_SEPARATOR_RE = re.compile(r"^\s{0,3}(?:(?:\*[ \t]*){3,}|(?:_[ \t]*){3,}|(?:-[ \t]*){3,}|=+)\s*$")
_MARKDOWN_TABLE_DELIMITER_CELL_RE = re.compile(r"^:?-{3,}:?$")
_MARKDOWN_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_MARKDOWN_TAB_STOP = 4


@dataclass(frozen=True)
class LintFinding:
    id: str
    severity: str
    file: str
    message: str
    hint: str
    line: Optional[int] = None
    code: Optional[str] = None
    skill: Optional[str] = None
    global_finding: bool = False

    def as_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "severity": self.severity,
            "file": self.file,
            "message": self.message,
            "hint": self.hint,
        }
        if self.line is not None:
            data["line"] = self.line
        if self.code is not None:
            data["code"] = self.code
        if self.skill is not None:
            data["skill"] = self.skill
        if self.global_finding:
            data["global"] = True
        return data


@dataclass
class SkillRecord:
    name: str
    skill_dir: Path
    skill_file: Path
    metadata: dict[str, Any]
    text: str
    body: str
    evals: list[dict[str, Any]]
    # Eval content is co-located under skill_dir/evals as evaluation metadata,
    # not runtime guidance. evals_dir is the suite root and evals_path its JSON.
    evals_dir: Path
    evals_path: Path
    evals_error: Optional[str]

    @property
    def public(self) -> bool:
        status = str(skill_metadata(self.metadata).get("status", "public")).strip().lower()
        return status not in PUBLIC_EXEMPT_STATUS

    @cached_property
    def validation(self) -> SkillValidationResult:
        # Computed lazily (once) so scoped runs that never consume it — e.g.
        # checks=[skill-md-size-lint] — keep the loader's bounded-read behavior
        # instead of validate_skill_dir's unbounded SKILL.md parse.
        return validate_skill_dir(self.skill_dir)

    @cached_property
    def has_helper_tests(self) -> bool:
        return _skill_has_helper_tests(self.skill_dir)


@dataclass
class LintContext:
    skills_root: Path
    max_skill_md_lines: int
    records: list[SkillRecord]
    findings: list[LintFinding]
    skipped_checks: list[dict[str, str]]


def run_v1_lints(
    skills_root: Path | str = "skills",
    *,
    evals_root: Path | str | None = None,
    checks: Optional[Iterable[str]] = None,
    max_skill_md_lines: int = SKILL_MD_MAX_LINES,
) -> dict[str, Any]:
    """Run deterministic v1 admission lints and return structured findings."""
    result, _records = _run_v1_lints_with_records(
        skills_root,
        evals_root=evals_root,
        checks=checks,
        max_skill_md_lines=max_skill_md_lines,
    )
    return result


def _run_v1_lints_with_records(
    skills_root: Path | str = "skills",
    *,
    evals_root: Path | str | None = None,
    checks: Optional[Iterable[str]] = None,
    max_skill_md_lines: int = SKILL_MD_MAX_LINES,
) -> tuple[dict[str, Any], list[SkillRecord]]:
    selected = tuple(checks or V1_LINT_IDS)
    unknown = sorted(set(selected).difference(V1_LINT_IDS))
    if unknown:
        raise ValueError(f"unknown agent skill lint check(s): {', '.join(unknown)}")

    root = Path(skills_root)
    # By default, each suite is co-located at skills/<skill>/evals/. An explicit
    # evals_root remains supported for isolated tooling tests and alternate QA
    # inputs, where suites use <evals_root>/<skill>/evals.json.
    evals_root_path = Path(evals_root) if evals_root is not None else None
    findings: list[LintFinding] = []
    records = _load_skill_records(root, evals_root_path, findings)
    context = LintContext(
        skills_root=root,
        max_skill_md_lines=max_skill_md_lines,
        records=records,
        findings=findings,
        skipped_checks=[],
    )
    root_error_codes = {"skills-root-missing", "skills-root-not-directory"}
    if not any(finding.global_finding and finding.code in root_error_codes for finding in findings):
        for check in selected:
            _LINT_FUNCTIONS[check](context)

    summary = _summary_from_severities((finding.severity for finding in context.findings), len(records))
    status = "failed" if summary["error_count"] else "ok"
    return {
        "schema_version": "1",
        "status": status,
        "passed": status == "ok",
        "skills_root": str(root),
        "checks": list(selected),
        "skipped_checks": context.skipped_checks,
        "summary": summary,
        "findings": [finding.as_dict() for finding in context.findings],
    }, records


def validate_skills(
    skills_root: Path | str = "skills",
    *,
    evals_root: Path | str | None = None,
    skill_name: Optional[str] = None,
    max_skill_md_lines: int = SKILL_MD_MAX_LINES,
) -> dict[str, Any]:
    """Compatibility wrapper for callers that validate one skill source root."""
    result, records = _run_v1_lints_with_records(
        skills_root,
        evals_root=evals_root,
        max_skill_md_lines=max_skill_md_lines,
    )

    if skill_name is not None:
        result["requested_skill"] = skill_name
        result["findings"] = [
            finding for finding in result["findings"] if _finding_matches_requested_skill(finding, skill_name)
        ]
        result["summary"] = _summary_from_severities(
            (finding.get("severity", FINDING_ERROR) for finding in result["findings"]),
            _matching_skill_count(records, skill_name),
        )
        result["status"] = "failed" if result["summary"]["error_count"] else "ok"
        result["passed"] = result["status"] == "ok"
    else:
        result["requested_skill"] = None
    return result


def _summary_from_severities(severities: Iterable[str], skill_count: int) -> dict[str, int]:
    severity_counts = Counter(severities)
    return {
        "skill_count": skill_count,
        "finding_count": sum(severity_counts.values()),
        "error_count": severity_counts.get(FINDING_ERROR, 0),
        "warning_count": severity_counts.get(FINDING_WARNING, 0),
        "info_count": severity_counts.get(FINDING_INFO, 0),
    }


def _finding_matches_requested_skill(finding: dict[str, Any], skill_name: str) -> bool:
    finding_skill = finding.get("skill")
    return finding_skill == skill_name or (finding_skill is None and finding.get("global") is True)


def _load_skill_records(skills_root: Path, evals_root: Path | None, findings: list[LintFinding]) -> list[SkillRecord]:
    if not skills_root.exists():
        findings.append(
            _finding(
                LINT_SKILL_FRONTMATTER,
                FINDING_ERROR,
                skills_root,
                "skills root does not exist",
                "Pass --skills-root pointing at the repository skills/ directory.",
                code="skills-root-missing",
                global_finding=True,
            )
        )
        return []
    if not skills_root.is_dir():
        findings.append(
            _finding(
                LINT_SKILL_FRONTMATTER,
                FINDING_ERROR,
                skills_root,
                "skills root is not a directory",
                "Pass --skills-root pointing at the repository skills/ directory.",
                code="skills-root-not-directory",
                global_finding=True,
            )
        )
        return []

    records = []
    for child in sorted(skills_root.iterdir(), key=lambda p: p.name):
        if should_skip_skill_dir(child):
            continue
        skill_file = child / SKILL_FILE_NAME
        text = _read_bounded_text(skill_file) if skill_file.is_file() else None
        metadata = _try_parse_frontmatter(skill_file) if text is not None else {}
        text = text or ""
        skill_name = str(metadata.get("name") or child.name)
        # By default, evals are co-located at <skill>/evals. An explicit external
        # eval root is supported for isolated tooling and uses one dir per skill.
        if evals_root is None:
            evals_dir = child / "evals"
        else:
            # The frontmatter name is attacker-controlled; only trust it for
            # the external-root path when it matches SKILL_NAME_RE. The invalid
            # name is still reported separately by the frontmatter lint.
            eval_dir_name = skill_name if SKILL_NAME_RE.match(skill_name) else child.name
            evals_dir = evals_root / eval_dir_name
        evals_path = evals_dir / "evals.json"
        evals, evals_error = _load_evals(evals_path)
        records.append(
            SkillRecord(
                name=skill_name,
                skill_dir=child,
                skill_file=skill_file,
                metadata=metadata,
                text=text,
                body=_skill_body(text),
                evals=evals,
                evals_dir=evals_dir,
                evals_path=evals_path,
                evals_error=evals_error,
            )
        )
    return records


def _matching_skill_count(records: list[SkillRecord], skill_name: str) -> int:
    return sum(1 for record in records if record.name == skill_name)


def _lint_frontmatter(context: LintContext) -> None:
    for record in context.records:
        for issue in record.validation.issues:
            context.findings.append(
                _finding(
                    LINT_SKILL_FRONTMATTER,
                    FINDING_ERROR,
                    Path(issue.path),
                    issue.message,
                    "Fix SKILL.md frontmatter before publishing this skill.",
                    code=issue.code,
                    skill=record.name,
                    line=_line_for_frontmatter_issue(record.skill_file, issue.code, issue.message),
                )
            )

        if record.public and _has_valid_name(record.metadata) and not record.name.startswith("nvflare-"):
            context.findings.append(
                _finding(
                    LINT_SKILL_FRONTMATTER,
                    FINDING_ERROR,
                    record.skill_file,
                    f"public NVFLARE skill name '{record.name}' must start with 'nvflare-'",
                    "Rename the skill directory and frontmatter name, or mark the skill draft/internal.",
                    code="skill-name-prefix-required",
                    skill=record.name,
                    line=_line_for_field(record.skill_file, "name"),
                )
            )


def _has_valid_name(metadata: dict[str, Any]) -> bool:
    name = metadata.get("name")
    return isinstance(name, str) and bool(name.strip())


def _lint_md_size(context: LintContext) -> None:
    for record in context.records:
        if not record.skill_file.is_file():
            continue
        if _is_oversized_text_file(record.skill_file):
            if _has_bounded_size_exception(record.skill_file):
                continue
            context.findings.append(
                _finding(
                    LINT_SKILL_MD_SIZE,
                    FINDING_ERROR,
                    record.skill_file,
                    f"SKILL.md exceeds the readable size limit of {MAX_SKILL_TEXT_FILE_BYTES} bytes",
                    "Move detailed workflow notes into references/ or add an approved exception marker near the top.",
                    code="skill-md-too-large",
                    skill=record.name,
                    line=1,
                )
            )
            continue
        lines = record.text.splitlines()
        max_lines = context.max_skill_md_lines
        if len(lines) > max_lines and not _has_size_exception(record.text):
            context.findings.append(
                _finding(
                    LINT_SKILL_MD_SIZE,
                    FINDING_ERROR,
                    record.skill_file,
                    f"SKILL.md has {len(lines)} lines; v1 hard limit is {max_lines}",
                    "Move detailed workflow notes into references/ or add an approved exception marker.",
                    code="skill-md-too-large",
                    skill=record.name,
                    line=max_lines + 1,
                )
            )
        word_count = len(record.text.split())
        if word_count > SKILL_MD_ADVISORY_WORDS:
            context.findings.append(
                _finding(
                    LINT_SKILL_MD_SIZE,
                    FINDING_INFO,
                    record.skill_file,
                    f"SKILL.md has about {word_count} whitespace-delimited tokens",
                    "The roughly 2,000-token target is advisory; keep SKILL.md concise when practical.",
                    code="skill-md-token-advisory",
                    skill=record.name,
                )
            )


def _lint_trigger(context: LintContext) -> None:
    for record in _public_records(context.records):
        searchable = f"{record.metadata.get('description', '')}\n{record.body}".lower()
        if not any(term in searchable for term in _TRIGGER_TERMS):
            context.findings.append(
                _finding(
                    LINT_SKILL_TRIGGER,
                    FINDING_ERROR,
                    record.skill_file,
                    "skill is missing trigger or use-boundary text",
                    "Add concise trigger guidance and negative boundary language to SKILL.md.",
                    code="skill-trigger-text-missing",
                    skill=record.name,
                )
            )

        if not _evals_available(
            context,
            LINT_SKILL_TRIGGER,
            record,
            "evals/evals.json is required for public skill trigger checks",
            "Add a guide-compatible evals/evals.json with positive and adjacent negative trigger evals.",
        ):
            continue
        if not any(_is_positive_eval(item, record.name) for item in record.evals):
            context.findings.append(
                _finding(
                    LINT_SKILL_TRIGGER,
                    FINDING_ERROR,
                    record.evals_path,
                    "missing positive trigger eval for this skill",
                    "Add an eval whose nvflare.expected_skill matches this skill.",
                    code="skill-positive-trigger-eval-missing",
                    skill=record.name,
                )
            )
        if not any(_is_adjacent_negative_eval(item, record.name) for item in record.evals):
            context.findings.append(
                _finding(
                    LINT_SKILL_TRIGGER,
                    FINDING_ERROR,
                    record.evals_path,
                    "missing adjacent negative trigger eval for this skill",
                    "Add an eval whose nvflare.negative_for names this skill and expected_skill names the neighbor.",
                    code="skill-adjacent-negative-eval-missing",
                    skill=record.name,
                )
            )


def _lint_trigger_overlap(context: LintContext) -> None:
    grouped: dict[str, list[SkillRecord]] = defaultdict(list)
    for record in _public_records(context.records):
        group = _trigger_overlap_group(record.name)
        grouped[group].append(record)

    max_trigger_overlap_skills = _max_trigger_overlap_skills()
    for group, records in grouped.items():
        if len(records) > max_trigger_overlap_skills:
            _skip(
                context,
                LINT_SKILL_TRIGGER_OVERLAP,
                f"group {group!r} has {len(records)} skills; limit is {max_trigger_overlap_skills}",
            )
            continue
        token_cache = {record.name: _trigger_tokens(record) for record in records}
        for i, left in enumerate(records):
            for right in records[i + 1 :]:
                if not _records_overlap(left, right, token_cache):
                    continue
                if _has_boundary_text(left) and _has_boundary_text(right) and _has_adjacent_negative_pair(left, right):
                    continue
                context.findings.append(
                    _finding(
                        LINT_SKILL_TRIGGER_OVERLAP,
                        FINDING_ERROR,
                        left.skill_file,
                        f"same trigger-group skills '{left.name}' and '{right.name}' have overlapping trigger language",
                        "Add use/do-not-use boundaries and adjacent negative evals covering the overlap.",
                        code="skill-trigger-overlap",
                        skill=left.name,
                    )
                )


def _trigger_overlap_group(skill_name: str) -> str:
    normalized = skill_name.strip().lower()
    if normalized.startswith("nvflare-"):
        normalized = normalized[len("nvflare-") :]
    family = normalized.split("-", maxsplit=1)[0].strip()
    if family:
        return f"nvflare-{family}"
    return skill_name.strip().lower() or skill_name


def _max_trigger_overlap_skills() -> int:
    value = os.environ.get("NVFLARE_AGENT_MAX_TRIGGER_OVERLAP_SKILLS")
    if value is None or value == "":
        return DEFAULT_MAX_TRIGGER_OVERLAP_SKILLS
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_MAX_TRIGGER_OVERLAP_SKILLS
    return parsed if parsed > 0 else DEFAULT_MAX_TRIGGER_OVERLAP_SKILLS


def _lint_global_negative(context: LintContext) -> None:
    for record in _public_records(context.records):
        if not _evals_available(
            context,
            LINT_SKILL_GLOBAL_NEGATIVE,
            record,
            "evals/evals.json is required for global negative coverage",
            "Add at least one eval representing an unrelated prompt that should trigger no FLARE skill.",
        ):
            continue
        if not any(_is_global_negative_eval(item) for item in record.evals):
            context.findings.append(
                _finding(
                    LINT_SKILL_GLOBAL_NEGATIVE,
                    FINDING_ERROR,
                    record.evals_path,
                    "missing global negative eval",
                    "Add an eval for an unrelated prompt with nvflare.expected_skill set to null or 'none'.",
                    code="skill-global-negative-eval-missing",
                    skill=record.name,
                )
            )


def _lint_policy_coverage(context: LintContext) -> None:
    for record in _public_records(context.records):
        has_behavior_ids = any(_behavior_id_count(item) for item in record.evals)
        has_helper_tests = record.has_helper_tests
        has_checklist = _skill_text_contains(record.skill_dir, "checklist")
        if has_behavior_ids or has_helper_tests or has_checklist:
            continue

        for file_path, text in _iter_skill_text_files(record.skill_dir):
            for line_no, line in enumerate(text.splitlines(), start=1):
                if _NORMATIVE_RE.search(line):
                    context.findings.append(
                        _finding(
                            LINT_SKILL_POLICY_COVERAGE,
                            FINDING_ERROR,
                            file_path,
                            "normative rule has no measurable behavior ID, helper test, or checklist coverage",
                            "Map required/prohibited behavior to evals/evals.json nvflare behavior IDs or tests.",
                            code="skill-policy-coverage-missing",
                            skill=record.name,
                            line=line_no,
                        )
                    )
                    break
            else:
                continue
            break


def _lint_process_metrics(context: LintContext) -> None:
    for record in _public_records(context.records):
        if not _evals_available(
            context,
            LINT_SKILL_PROCESS_METRIC,
            record,
            "evals/evals.json is required for process-metric coverage",
            "Add process metric contracts under nvflare.process_metrics.",
        ):
            continue

        process_metrics = []
        for item in record.evals:
            for metric in _process_metrics(item):
                process_metrics.append((item, metric))

        if not process_metrics:
            context.findings.append(
                _finding(
                    LINT_SKILL_PROCESS_METRIC,
                    FINDING_ERROR,
                    record.evals_path,
                    "missing process metric contracts for this public skill",
                    "Add nvflare.process_metrics entries for first-pass quality, correction count, unwanted actions, "
                    "validation evidence, or other skill-process outcomes.",
                    code="skill-process-metric-missing",
                    skill=record.name,
                )
            )
            continue

        for item, metric in process_metrics:
            if not isinstance(metric, dict):
                context.findings.append(
                    _finding(
                        LINT_SKILL_PROCESS_METRIC,
                        FINDING_ERROR,
                        record.evals_path,
                        f"eval '{item.get('id', '<missing>')}' process metric must be an object",
                        "Use objects with at least id and description fields.",
                        code="skill-process-metric-type",
                        skill=record.name,
                    )
                )
                continue
            metric_id = metric.get("id")
            description = metric.get("description")
            if not isinstance(metric_id, str) or not metric_id.strip():
                context.findings.append(
                    _finding(
                        LINT_SKILL_PROCESS_METRIC,
                        FINDING_ERROR,
                        record.evals_path,
                        f"eval '{item.get('id', '<missing>')}' process metric is missing id",
                        "Add a stable metric id such as turns_to_acceptable or user_correction_count.",
                        code="skill-process-metric-id-missing",
                        skill=record.name,
                    )
                )
            if not isinstance(description, str) or not description.strip():
                context.findings.append(
                    _finding(
                        LINT_SKILL_PROCESS_METRIC,
                        FINDING_ERROR,
                        record.evals_path,
                        f"eval '{item.get('id', '<missing>')}' process metric '{metric_id}' is missing description",
                        "Describe what the metric measures and what evidence records it.",
                        code="skill-process-metric-description-missing",
                        skill=record.name,
                    )
                )


def _lint_command_drift(context: LintContext) -> None:
    for record in _public_records(context.records):
        for file_path, text in _iter_skill_text_files(record.skill_dir, include_scripts=True):
            for line_no, command in _extract_nvflare_commands(text):
                message = _command_drift_message(command)
                if message is None:
                    continue
                context.findings.append(
                    _finding(
                        LINT_SKILL_COMMAND_DRIFT,
                        FINDING_ERROR,
                        file_path,
                        message,
                        "Update the referenced nvflare command or the CLI command schema before publishing.",
                        code="skill-command-drift",
                        skill=record.name,
                        line=line_no,
                    )
                )


def _lint_helper_scripts(context: LintContext) -> None:
    for record in _public_records(context.records):
        scripts_dir = record.skill_dir / "scripts"
        if not scripts_dir.is_dir():
            continue
        script_files = list(_iter_files_no_follow(scripts_dir))
        if script_files and not record.has_helper_tests:
            context.findings.append(
                _finding(
                    LINT_SKILL_HELPER_SCRIPT,
                    FINDING_ERROR,
                    scripts_dir,
                    "helper scripts are shipped without tests",
                    "Add tests under the skill or repository test tree for each shipped helper script.",
                    code="skill-helper-tests-missing",
                    skill=record.name,
                )
            )

        for script in script_files:
            try:
                if script.stat().st_size > MAX_SKILL_TEXT_FILE_BYTES:
                    continue
            except OSError:
                continue
            text = script.read_text(encoding="utf-8", errors="replace")
            lowered = text.lower()
            if "promoted_to:" in lowered or "_promoted_to:" in lowered:
                context.findings.append(
                    _finding(
                        LINT_SKILL_HELPER_SCRIPT,
                        FINDING_ERROR,
                        script,
                        "helper script is marked as promoted to a product CLI command",
                        "Update SKILL.md to call the product CLI command instead of the promoted helper script.",
                        code="skill-helper-promoted",
                        skill=record.name,
                    )
                )
            declares_json_output = any(
                token in lowered for token in ("json output", "stdout json", "json stdout", "jsonl", "machine-readable")
            )
            if script.suffix == ".py" and declares_json_output and "json.dumps" not in text and "json.dump" not in text:
                context.findings.append(
                    _finding(
                        LINT_SKILL_HELPER_SCRIPT,
                        FINDING_WARNING,
                        script,
                        "script mentions JSON but does not appear to write JSON with the json module",
                        "Ensure machine-readable stdout is valid JSON and diagnostics go to stderr.",
                        code="skill-helper-json-unclear",
                        skill=record.name,
                    )
                )


def _lint_fixtures(context: LintContext) -> None:
    for record in _public_records(context.records):
        if record.evals_error:
            _add_evals_error(context, LINT_SKILL_FIXTURE, record)
            continue
        if not record.evals_path.is_file():
            continue

        for item in record.evals:
            files = item.get("files", [])
            if files is None:
                files = []
            if not isinstance(files, list):
                context.findings.append(
                    _finding(
                        LINT_SKILL_FIXTURE,
                        FINDING_ERROR,
                        record.evals_path,
                        f"eval '{item.get('id', '<missing>')}' files field must be a list",
                        "Use guide-compatible files: [...] entries relative to the skill directory.",
                        code="skill-fixture-files-type",
                        skill=record.name,
                    )
                )
                continue

            if _eval_mentions_file_editing(item) and not files:
                context.findings.append(
                    _finding(
                        LINT_SKILL_FIXTURE,
                        FINDING_ERROR,
                        record.evals_path,
                        f"eval '{item.get('id', '<missing>')}' describes file editing without input fixtures",
                        "Add deterministic input files under evals/files/ and reference them from the eval.",
                        code="skill-fixture-input-missing",
                        skill=record.name,
                    )
                )

            for rel_path in files:
                fixture_path = record.evals_dir / str(rel_path)
                resolved_fixture_path = fixture_path.resolve(strict=False)
                resolved_evals_dir = record.evals_dir.resolve()
                if not resolved_fixture_path.is_relative_to(resolved_evals_dir):
                    context.findings.append(
                        _finding(
                            LINT_SKILL_FIXTURE,
                            FINDING_ERROR,
                            record.evals_path,
                            f"eval fixture path escapes eval suite directory: {rel_path}",
                            "Use fixture paths relative to the eval suite directory.",
                            code="skill-fixture-path-escape",
                            skill=record.name,
                        )
                    )
                    continue
                # A fixture may be one file or a dataset directory (e.g. an
                # image folder); a directory must contain at least one file.
                if not fixture_path.is_file() and not (fixture_path.is_dir() and _has_files(fixture_path)):
                    context.findings.append(
                        _finding(
                            LINT_SKILL_FIXTURE,
                            FINDING_ERROR,
                            record.evals_path,
                            f"eval fixture does not exist: {rel_path}",
                            "Place deterministic fixtures under evals/files/ and reference existing files or "
                            "non-empty dataset directories.",
                            code="skill-fixture-file-missing",
                            skill=record.name,
                        )
                    )

        files_dir = record.evals_dir / "files"
        if _has_files(files_dir) and not _has_fixture_notes(record.evals_dir):
            context.findings.append(
                _finding(
                    LINT_SKILL_FIXTURE,
                    FINDING_ERROR,
                    files_dir,
                    "eval fixtures are missing source/provenance notes",
                    "Add README.md, files/README.md, or files/SOURCE.md in the eval suite dir.",
                    code="skill-fixture-notes-missing",
                    skill=record.name,
                )
            )


_DESIGN_DOC_REF_RE = re.compile(r"docs[\\/]+design\b", re.IGNORECASE)
_EVALUATOR_HOOK_RE = re.compile(
    r"(?:"
    r"\bevals?/"
    r"|\bevals\.json\b"
    r"|\beval\s+cases?\b"
    r"|\beval\s+fixtures?\b"
    r"|\bevaluator\b"
    r"|\bbenchmark[ -]?harness\b"  # benchmark-harness instructions (not "benchmark dataset")
    r"|\beval\s+(?:suite|harness)\b"  # evaluator harness references
    r"|\bgrader\b"  # eval grader references
    r"|\beval\s*=\s*\w"  # eval-mode toggles such as eval=on
    r"|--eval\b"  # evaluator harness flags such as --eval / --eval-only
    r"|(?-i:\b[A-Z][A-Z0-9_]*_EVAL(?![A-Z])[A-Z0-9_]*\b)"  # env vars such as NVFLARE_SKILL_EVAL, not DEFAULT_EVALUATION
    r")",
    re.IGNORECASE,
)
# Names excluded from runtime-guidance scanning. ``evals`` is co-located
# evaluation metadata, which the standard installer copies with the complete
# skill directory but which is not runtime guidance. Byte-code artifacts and
# ``__pycache__`` are likewise not meaningful skill guidance. This controls
# lint traversal only; it does not configure the external skills installer.
SKILL_RUNTIME_GUIDANCE_EXCLUDE_NAMES = frozenset({"__pycache__", "*.pyc", "*.pyo", "evals"})
# Directory-name subset of SKILL_RUNTIME_GUIDANCE_EXCLUDE_NAMES (byte-code file
# globs are not directory names). The runtime-boundary lint uses this to prune
# its guidance scan.
_RUNTIME_BOUNDARY_EXCLUDED_DIRS = {name for name in SKILL_RUNTIME_GUIDANCE_EXCLUDE_NAMES if not name.startswith("*")}
_RUNTIME_TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".rst",
    ".py",
    ".sh",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".j2",
    ".jinja",
    ".jinja2",
    "",
}


def _lint_runtime_boundary(context: LintContext) -> None:
    """Packaged runtime skill content must stay inside the runtime boundary.

    Runtime guidance is the skill content used to direct an agent. A top-level
    ``evals/`` suite is allowed as co-located evaluation metadata and is omitted
    from this guidance scan. Nested ``evals/`` directories are invalid because
    the supported layout is ``<skill>/evals/``. Runtime guidance must not
    reference ``docs/design/`` documents or contain evaluator hooks or
    benchmark-harness-only instructions. The scan covers every skill record
    (public and non-public) and every shared reference directory, not only
    ``SKILL.md`` and ``.md`` references.
    """
    for record in context.records:
        for eval_dir in _iter_misplaced_eval_dirs(record.skill_dir):
            context.findings.append(
                _finding(
                    LINT_SKILL_RUNTIME_BOUNDARY,
                    FINDING_ERROR,
                    eval_dir,
                    "eval suite must be the top-level evals/ directory of its skill",
                    "Move the eval suite to skills/<skill>/evals/, the supported co-located eval location.",
                    code="skill-runtime-eval-dir-in-skill",
                    skill=record.name,
                )
            )
        for file_path, text in _iter_packaged_runtime_files(record.skill_dir):
            _scan_runtime_boundary(context, file_path, text, skill=record.name)


def _lint_dependency_install_safety(context: LintContext) -> None:
    """Reject runtime guidance that suppresses dependency review or consent.

    NVSkillsEvaluator's keyless Tier 1 security checks cover structural and
    code-security concerns, but they do not interpret dependency-install
    authorization policy. Keep this repository-owned check deterministic and
    limited to runtime guidance (SKILL.md and references), excluding eval
    prompts and fixtures that intentionally contain adversarial instructions.
    """
    for record in context.records:
        for file_path in _iter_packaged_runtime_text_paths(record.skill_dir):
            try:
                text = read_regular_text_file(file_path, max_bytes=MAX_SKILL_TEXT_FILE_BYTES, errors="replace")
            except RegularFileTooLargeError:
                context.findings.append(
                    _finding(
                        LINT_SKILL_DEPENDENCY_INSTALL_SAFETY,
                        FINDING_ERROR,
                        file_path,
                        "packaged runtime guidance exceeds the dependency-policy scan limit",
                        "Split the guidance into bounded files so every dependency instruction is scanned.",
                        code="dependency-install-guidance-too-large",
                        skill=record.name,
                    )
                )
                continue
            except (OSError, UnicodeError, ValueError):
                continue
            blocks = list(_iter_markdown_policy_blocks(text))
            fenced_line_numbers = _markdown_fenced_line_numbers(text)
            # Markdown may put the action and a bare bypass in adjacent list
            # items, headings, or quoted blocks. Pair only that semantic shape;
            # arbitrary neighboring statements remain independent.
            scan_blocks = list(blocks)
            for left, right in zip(blocks, blocks[1:]):
                if left[0] in fenced_line_numbers or right[0] in fenced_line_numbers:
                    continue
                left_text = left[1]
                right_text = right[1]
                if (_DEPENDENCY_INSTALL_TERMS_RE.search(left_text) and _is_bare_confirmation_bypass(right_text)) or (
                    _is_bare_confirmation_bypass(left_text) and _DEPENDENCY_INSTALL_TERMS_RE.search(right_text)
                ):
                    scan_blocks.append((left[0], f"{left_text} {right_text}"))
            seen_findings = set()
            for line_number, paragraph in scan_blocks:
                if not _DEPENDENCY_INSTALL_TERMS_RE.search(paragraph):
                    continue
                if _has_dependency_confirmation_bypass(paragraph):
                    finding_key = (file_path, line_number, "dependency-install-confirmation-bypass")
                    if finding_key not in seen_findings:
                        seen_findings.add(finding_key)
                        context.findings.append(
                            _finding(
                                LINT_SKILL_DEPENDENCY_INSTALL_SAFETY,
                                FINDING_ERROR,
                                file_path,
                                "dependency-install guidance suppresses user confirmation",
                                "Show a redacted install plan and require confirmation unless the user "
                                "explicitly requested unattended installation.",
                                code="dependency-install-confirmation-bypass",
                                skill=record.name,
                                line=line_number,
                            )
                        )
                if _has_dependency_review_bypass(paragraph):
                    finding_key = (file_path, line_number, "dependency-install-review-bypass")
                    if finding_key not in seen_findings:
                        seen_findings.add(finding_key)
                        context.findings.append(
                            _finding(
                                LINT_SKILL_DEPENDENCY_INSTALL_SAFETY,
                                FINDING_ERROR,
                                file_path,
                                "dependency-install guidance suppresses package or source review",
                                "Require static review and flag suspicious package names, sources, "
                                "credentials, indexes, and installer options.",
                                code="dependency-install-review-bypass",
                                skill=record.name,
                                line=line_number,
                            )
                        )


# Canonical lint registry: single source of truth for lint IDs, their run
# order, and their implementations. V1_LINT_IDS and _LINT_FUNCTIONS derive
# from it; do not maintain separate lists.
_LINT_REGISTRY = (
    (LINT_SKILL_FRONTMATTER, _lint_frontmatter),
    (LINT_SKILL_MD_SIZE, _lint_md_size),
    (LINT_SKILL_TRIGGER, _lint_trigger),
    (LINT_SKILL_TRIGGER_OVERLAP, _lint_trigger_overlap),
    (LINT_SKILL_GLOBAL_NEGATIVE, _lint_global_negative),
    (LINT_SKILL_POLICY_COVERAGE, _lint_policy_coverage),
    (LINT_SKILL_PROCESS_METRIC, _lint_process_metrics),
    (LINT_SKILL_COMMAND_DRIFT, _lint_command_drift),
    (LINT_SKILL_HELPER_SCRIPT, _lint_helper_scripts),
    (LINT_SKILL_FIXTURE, _lint_fixtures),
    (LINT_SKILL_RUNTIME_BOUNDARY, _lint_runtime_boundary),
    (LINT_SKILL_DEPENDENCY_INSTALL_SAFETY, _lint_dependency_install_safety),
)
V1_LINT_IDS = tuple(lint_id for lint_id, _ in _LINT_REGISTRY)
_LINT_FUNCTIONS = dict(_LINT_REGISTRY)


def _iter_misplaced_eval_dirs(skill_dir: Path) -> Iterable[Path]:
    """Yield nested ``evals`` directories, excluding the allowed top-level suite.

    The top-level ``<skill>/evals`` location is the Agent Skills Specification
    location and is omitted from the runtime-guidance scan. A nested directory
    such as ``references/evals`` is outside the supported layout and is invalid.
    """
    if not skill_dir.is_dir():
        return
    excluded = _RUNTIME_BOUNDARY_EXCLUDED_DIRS - {"evals"}
    for current_dir, dir_names, _file_names in _walk_no_follow(skill_dir, excluded):
        if "evals" in dir_names:
            if current_dir != skill_dir:
                yield current_dir / "evals"
            dir_names.remove("evals")


def _iter_packaged_runtime_files(skill_dir: Path) -> Iterable[tuple[Path, str]]:
    """Yield decoded text files a skill ships as runtime content."""
    for path in _iter_packaged_runtime_text_paths(skill_dir):
        content = _read_runtime_text_file(path)
        if content is not None:
            yield path, content


def _iter_packaged_runtime_text_paths(skill_dir: Path) -> Iterable[Path]:
    """Yield every text-like packaged runtime path, excluding evaluation data."""
    if not skill_dir.is_dir():
        return
    for path in _iter_files_no_follow(skill_dir, excluded_dir_names=_RUNTIME_BOUNDARY_EXCLUDED_DIRS):
        if path.suffix.lower() in _RUNTIME_TEXT_SUFFIXES:
            yield path


def _read_runtime_text_file(path: Path) -> Optional[str]:
    # Runtime-scan-specific guards (skip symlinked files and non-text suffixes),
    # then defer the size-cap + bounded read to the shared reader.
    if path.is_symlink() or path.suffix.lower() not in _RUNTIME_TEXT_SUFFIXES:
        return None
    return _read_bounded_text(path)


def _scan_runtime_boundary(context: LintContext, file_path: Path, text: str, *, skill: Optional[str]) -> None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if _DESIGN_DOC_REF_RE.search(line):
            context.findings.append(
                _finding(
                    LINT_SKILL_RUNTIME_BOUNDARY,
                    FINDING_ERROR,
                    file_path,
                    "packaged runtime skill content references docs/design/ documents",
                    "Copy the runtime-relevant rule into SKILL.md/reference content or product docs; "
                    "design docs are authoring and review inputs.",
                    code="skill-runtime-design-doc-ref",
                    skill=skill,
                    line=line_no,
                    global_finding=skill is None,
                )
            )
        if _EVALUATOR_HOOK_RE.search(line):
            context.findings.append(
                _finding(
                    LINT_SKILL_RUNTIME_BOUNDARY,
                    FINDING_ERROR,
                    file_path,
                    "packaged runtime skill content contains evaluator or benchmark-harness instructions",
                    "Keep evaluator hooks and benchmark instructions in the co-located repo-only "
                    "evals/ content, not in SKILL.md, references/, scripts/, or shared references.",
                    code="skill-runtime-evaluator-hook",
                    skill=skill,
                    line=line_no,
                    global_finding=skill is None,
                )
            )


def _try_parse_frontmatter(skill_file: Path) -> dict[str, Any]:
    try:
        return parse_skill_frontmatter(skill_file)
    except Exception:
        return {}


def _skill_body(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[index + 1 :])
    return text


def _load_evals(evals_path: Path) -> tuple[list[dict[str, Any]], Optional[str]]:
    if not evals_path.is_file():
        return [], None
    if _is_oversized_text_file(evals_path):
        return [], f"evals.json exceeds size limit ({MAX_SKILL_TEXT_FILE_BYTES} bytes)"
    try:
        raw = json.loads(evals_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [], f"failed to parse evals.json: {e}"
    if isinstance(raw, dict):
        items = raw.get("evals", [])
    elif isinstance(raw, list):
        items = raw
    else:
        return [], "evals.json must be an object with an evals list or a list"
    if not isinstance(items, list):
        return [], "evals.json field 'evals' must be a list"
    evals = [item for item in items if isinstance(item, dict)]
    if len(evals) != len(items):
        return evals, "each evals.json entry must be an object"
    return evals, None


def _add_evals_error(context: LintContext, lint_id: str, record: SkillRecord) -> None:
    context.findings.append(
        _finding(
            lint_id,
            FINDING_ERROR,
            record.evals_path,
            record.evals_error or "evals.json is invalid",
            "Use guide-compatible JSON with an evals list.",
            code="skill-evals-invalid",
            skill=record.name,
        )
    )


def _evals_available(
    context: LintContext, lint_id: str, record: SkillRecord, missing_message: str, missing_hint: str
) -> bool:
    """Report invalid or missing evals.json for one lint; True when evals are usable."""
    if record.evals_error:
        _add_evals_error(context, lint_id, record)
        return False
    if not record.evals_path.is_file():
        context.findings.append(
            _finding(
                lint_id,
                FINDING_ERROR,
                record.evals_path,
                missing_message,
                missing_hint,
                code="skill-evals-missing",
                skill=record.name,
            )
        )
        return False
    return True


def _public_records(records: list[SkillRecord]) -> list[SkillRecord]:
    return [record for record in records if record.public]


def _is_positive_eval(item: dict[str, Any], skill_name: str) -> bool:
    nvflare = _nvflare_ext(item)
    if nvflare.get("expected_skill") == skill_name:
        return True
    tags = _eval_tags(item)
    return "positive" in tags or "positive-trigger" in tags


def _is_adjacent_negative_eval(item: dict[str, Any], skill_name: str) -> bool:
    nvflare = _nvflare_ext(item)
    if nvflare.get("negative_for") == skill_name:
        return True
    tags = _eval_tags(item)
    return "adjacent-negative" in tags or "adjacent_negative" in tags


def _is_global_negative_eval(item: dict[str, Any]) -> bool:
    nvflare = _nvflare_ext(item)
    expected_skill = nvflare.get("expected_skill")
    if expected_skill is None and "expected_skill" in nvflare:
        return True
    if isinstance(expected_skill, str) and expected_skill.lower() in {"none", "no-skill", "no_skill"}:
        return True
    if nvflare.get("negative_for") == "*":
        return True
    tags = _eval_tags(item)
    text = _eval_text(item).lower()
    return "global-negative" in tags or "global_negative" in tags or "trigger no flare skill" in text


def _eval_tags(item: dict[str, Any]) -> set[str]:
    values = item.get("tags", [])
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return set()
    return {str(value).strip().lower() for value in values}


def _nvflare_ext(item: dict[str, Any]) -> dict[str, Any]:
    nvflare = item.get("nvflare", {})
    return nvflare if isinstance(nvflare, dict) else {}


def _behavior_id_count(item: dict[str, Any]) -> int:
    nvflare = _nvflare_ext(item)
    count = 0
    for key in ("mandatory_behavior", "optional_behavior", "prohibited_behavior"):
        values = nvflare.get(key, [])
        if isinstance(values, list):
            count += len(values)
    return count


def _process_metrics(item: dict[str, Any]) -> list[Any]:
    nvflare = _nvflare_ext(item)
    metrics = nvflare.get("process_metrics", [])
    return metrics if isinstance(metrics, list) else [metrics]


def _eval_text(item: dict[str, Any]) -> str:
    parts = [str(item.get("id", "")), str(item.get("prompt", "")), str(item.get("expected_output", ""))]
    assertions = item.get("assertions", [])
    if isinstance(assertions, list):
        parts.extend(str(assertion) for assertion in assertions)
    return "\n".join(parts)


def _records_overlap(left: SkillRecord, right: SkillRecord, token_cache: dict[str, set[str]]) -> bool:
    left_tokens = token_cache[left.name]
    right_tokens = token_cache[right.name]
    if not left_tokens or not right_tokens:
        return False
    shared = left_tokens.intersection(right_tokens)
    smaller = min(len(left_tokens), len(right_tokens))
    return len(shared) >= 4 and (len(shared) / smaller) >= 0.35


def _trigger_tokens(record: SkillRecord) -> set[str]:
    prompts = "\n".join(str(item.get("prompt", "")) for item in record.evals)
    text = f"{record.metadata.get('description', '')}\n{prompts}"
    return {
        token
        for token in _SIGNIFICANT_TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and not token.startswith("nvflare")
    }


def _has_boundary_text(record: SkillRecord) -> bool:
    text = f"{record.metadata.get('description', '')}\n{record.body}".lower()
    return any(term in text for term in _BOUNDARY_TERMS)


def _has_adjacent_negative_pair(left: SkillRecord, right: SkillRecord) -> bool:
    return any(_negative_for_neighbor(item, left.name, right.name) for item in left.evals) or any(
        _negative_for_neighbor(item, right.name, left.name) for item in right.evals
    )


def _negative_for_neighbor(item: dict[str, Any], skill_name: str, neighbor_name: str) -> bool:
    nvflare = _nvflare_ext(item)
    return nvflare.get("negative_for") == skill_name and nvflare.get("expected_skill") == neighbor_name


def _strip_backticks(value: str) -> str:
    return value.strip().strip("`")


def _extract_nvflare_commands(text: str) -> list[tuple[int, str]]:
    commands = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        matches = list(_BACKTICK_NVFLARE_RE.finditer(line))
        if matches:
            commands.extend((line_no, match.group(1).strip()) for match in matches)
            continue
        index = line.find("nvflare ")
        if index == -1:
            continue
        commands.append((line_no, _trim_command(line[index:])))
    return commands


def _trim_command(text: str) -> str:
    text = text.strip()
    for separator in ("&&", "|", ";"):
        if separator in text:
            text = text.split(separator, 1)[0].strip()
    return text


def _command_drift_message(command: str) -> Optional[str]:
    tokens = _command_tokens(command)
    if not tokens or tokens[0] != "nvflare":
        return None
    positional = [token for token in tokens[1:] if not token.startswith("-") and not _looks_like_value(token)]
    if not positional:
        return None
    root = positional[0]
    if root not in _KNOWN_NVFLARE_ROOT_COMMANDS:
        return f"unknown nvflare command root '{root}' in '{command}'"
    if root != "agent":
        return None

    if len(positional) >= 2 and positional[1] not in _KNOWN_AGENT_COMMANDS:
        return f"unknown nvflare agent command '{positional[1]}' in '{command}'"

    if positional[1:2] == ["inspect"]:
        if len(positional) < 3:
            if "--schema" not in tokens:
                return f"nvflare agent inspect requires a source or data capability in '{command}'"
            command_key = "agent inspect"
        elif positional[2] not in _KNOWN_AGENT_INSPECT_CAPABILITIES:
            return f"unknown nvflare agent inspect capability '{positional[2]}' in '{command}'"
        else:
            command_key = " ".join(positional[:3])
    else:
        command_key = " ".join(positional[:2])
    allowed_flags = _KNOWN_AGENT_FLAGS.get(command_key, _KNOWN_AGENT_FLAGS.get(root, set()))
    for token in tokens:
        if token.startswith("--"):
            flag = token.split("=", 1)[0]
            if flag not in allowed_flags:
                return f"unknown flag '{flag}' for 'nvflare {command_key}' in '{command}'"
    return None


def _command_tokens(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        # shlex.split raises on an unbalanced quote (e.g. an apostrophe in prose
        # like "the server's state"). Falling back to a plain whitespace split
        # keeps the drift lint fail-closed: the "nvflare <subcommand>" tokens are
        # still extracted and checked instead of the command silently passing.
        tokens = command.split()
    try:
        start = tokens.index("nvflare")
    except ValueError:
        return []
    command_tokens = []
    for token in tokens[start:]:
        if token in {"&&", "|", ";"}:
            break
        if not _safe_command_token(token):
            break
        command_tokens.append(token)
    return command_tokens


def _safe_command_token(token: str) -> bool:
    return bool(_SAFE_COMMAND_TOKEN_RE.match(token))


def _looks_like_value(token: str) -> bool:
    return token.startswith("<") or "/" in token or token in {"on", "off", "json", "jsonl", "human"}


def _skill_has_helper_tests(skill_dir: Path) -> bool:
    tests_dir = skill_dir / "tests"
    if tests_dir.is_dir() and any(True for _path in _iter_files_no_follow(tests_dir)):
        return True
    return any(path.name.endswith(("_test.py", ".test.py")) for path in _iter_files_no_follow(skill_dir))


def _skill_text_contains(skill_dir: Path, needle: str) -> bool:
    needle = needle.lower()
    return any(needle in text.lower() for _, text in _iter_skill_text_files(skill_dir))


def _iter_skill_text_files(skill_dir: Path, *, include_scripts: bool = False) -> Iterable[tuple[Path, str]]:
    candidates = [skill_dir / SKILL_FILE_NAME]
    references_dir = skill_dir / "references"
    if references_dir.is_dir():
        candidates.extend(
            path for path in _iter_files_no_follow(references_dir) if path.suffix.lower() in {".md", ".txt"}
        )
    if include_scripts:
        scripts_dir = skill_dir / "scripts"
        if scripts_dir.is_dir():
            candidates.extend(_iter_files_no_follow(scripts_dir))
    for path in candidates:
        # Guard the direct SKILL.md join too (and re-guard reference/script
        # candidates): never follow a symlink, whose target may be outside the
        # skill tree.
        if path.is_file() and not path.is_symlink():
            try:
                if path.stat().st_size > MAX_SKILL_TEXT_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path, path.read_text(encoding="utf-8", errors="replace")


def _iter_markdown_policy_blocks(text: str) -> Iterable[tuple[int, str]]:
    """Yield policy text without combining separate Markdown blocks.

    Wrapped prose, blockquote continuations, and list-item continuations remain
    searchable as one unit, while headings, separate blockquote statements,
    list items, table rows, and separators keep their Markdown boundaries.
    Fenced content joins wrapped lines of the same statement (the same
    continuation heuristic used for blockquotes) but still keeps distinct
    literal lines apart, so unrelated example lines cannot satisfy one matcher
    while a single instruction wrapped across lines still can.
    """
    lines = text.splitlines()
    table_row_numbers = _markdown_table_row_numbers(lines)
    block_lines = []
    block_start = 1
    block_kind = ""
    list_content_indent = 0
    list_blank_pending = False
    fence_marker = ""
    fenced_lines = []
    fenced_block_start = 1
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        fence_match = _markdown_fence_match(line)

        if fence_marker:
            if fence_match and fence_match[0] == fence_marker[0] and len(fence_match) >= len(fence_marker):
                if fenced_lines:
                    yield fenced_block_start, " ".join(fenced_lines)
                    fenced_lines = []
                fence_marker = ""
            elif stripped:
                content = _normalize_policy_text(_strip_markdown_quote_container(line).strip())
                if fenced_lines and _is_markdown_fenced_continuation(fenced_lines[-1], content):
                    fenced_lines.append(content)
                else:
                    if fenced_lines:
                        yield fenced_block_start, " ".join(fenced_lines)
                    fenced_lines = [content]
                    fenced_block_start = line_number
            else:
                if fenced_lines:
                    yield fenced_block_start, " ".join(fenced_lines)
                    fenced_lines = []
            continue

        if fence_match:
            if block_lines:
                yield block_start, " ".join(block_lines)
                block_lines = []
                block_kind = ""
                list_blank_pending = False
            fence_marker = fence_match
            continue

        if not stripped:
            if block_kind == "list":
                list_blank_pending = True
            elif block_lines:
                yield block_start, " ".join(block_lines)
                block_lines = []
                block_kind = ""
            continue

        if list_blank_pending:
            indentation = _markdown_leading_indent(line)
            if indentation >= list_content_indent:
                block_lines.append(_normalize_policy_text(stripped))
                list_blank_pending = False
                continue
            yield block_start, " ".join(block_lines)
            block_lines = []
            block_kind = ""
            list_blank_pending = False

        if _MARKDOWN_STRUCTURAL_SEPARATOR_RE.match(line):
            if block_lines:
                yield block_start, " ".join(block_lines)
                block_lines = []
                block_kind = ""
            yield line_number, _normalize_policy_text(stripped)
            continue

        blockquote_match = _MARKDOWN_BLOCKQUOTE_RE.match(line)
        if blockquote_match:
            content = _normalize_policy_text(line[blockquote_match.end() :].lstrip())
            if not content:
                if block_lines:
                    yield block_start, " ".join(block_lines)
                    block_lines = []
                    block_kind = ""
                continue
            if block_kind == "blockquote" and _is_markdown_blockquote_continuation(block_lines[-1], content):
                block_lines.append(content)
                continue
            if block_lines:
                yield block_start, " ".join(block_lines)
            block_lines = [content]
            block_start = line_number
            block_kind = "blockquote"
            continue

        list_item_match = _MARKDOWN_LIST_ITEM_RE.match(line)
        if list_item_match:
            if block_lines:
                yield block_start, " ".join(block_lines)
            block_lines = [_normalize_policy_text(line[list_item_match.end() :].strip())]
            block_start = line_number
            block_kind = "list"
            list_content_indent = _markdown_column_width(line[: list_item_match.end()])
            continue

        heading_match = _MARKDOWN_ATX_HEADING_RE.match(line)
        if heading_match or _MARKDOWN_TABLE_ROW_RE.match(line) or line_number in table_row_numbers:
            if block_lines:
                yield block_start, " ".join(block_lines)
                block_lines = []
                block_kind = ""
            content = line[heading_match.end() :] if heading_match else stripped
            yield line_number, _normalize_policy_text(content)
            continue

        if block_kind == "blockquote":
            if _is_markdown_blockquote_continuation(block_lines[-1], stripped):
                block_lines.append(_normalize_policy_text(stripped))
                continue
            yield block_start, " ".join(block_lines)
            block_lines = []
            block_kind = ""
        elif block_kind == "list":
            # A non-blank line directly following a list item (no blank line in
            # between) is a CommonMark "lazy continuation" of that item's paragraph
            # regardless of indentation -- it has already been checked above and is
            # not itself a new block (list item, blockquote, heading, separator, or
            # table row), so it keeps searching alongside the item's text.
            block_lines.append(_normalize_policy_text(stripped))
            continue
        if not block_lines:
            block_start = line_number
            block_kind = "paragraph"
        block_lines.append(_normalize_policy_text(stripped))

    if block_lines:
        yield block_start, " ".join(block_lines)
    if fenced_lines:
        yield fenced_block_start, " ".join(fenced_lines)


def _markdown_fence_match(line: str) -> str:
    """Return a fence marker after stripping quote/list containers."""
    candidate = _strip_markdown_quote_container(line)
    list_match = _MARKDOWN_LIST_ITEM_RE.match(candidate)
    if list_match:
        candidate = candidate[list_match.end() :]
    match = re.match(r"^\s*(`{3,}|~{3,})(?:[^`~]*)$", candidate)
    return match.group(1) if match else ""


def _strip_markdown_quote_container(line: str) -> str:
    match = _MARKDOWN_BLOCKQUOTE_RE.match(line)
    return line[match.end() :] if match else line


def _markdown_fenced_line_numbers(text: str) -> set[int]:
    """Return content-line numbers inside normalized Markdown fences."""
    result = set()
    marker = ""
    for line_number, line in enumerate(text.splitlines(), start=1):
        candidate = _markdown_fence_match(line)
        if candidate and (not marker or candidate[0] == marker[0] and len(candidate) >= len(marker)):
            marker = "" if marker else candidate
        elif marker:
            result.add(line_number)
    return result


def _normalize_policy_text(text: str) -> str:
    """Remove inline Markdown wrappers that are not part of policy meaning."""
    return re.sub(r"(?<!\w)(?:\*\*|__|`+)|(?:\*\*|__|`+)(?!\w)", "", text).strip()


def _markdown_table_row_numbers(lines: list[str]) -> set[int]:
    """Return one-based row numbers for pipe tables, including unbordered tables."""
    row_numbers = set()
    for delimiter_index, line in enumerate(lines):
        content = line.strip().removeprefix("|").removesuffix("|")
        cells = [cell.strip() for cell in content.split("|")]
        if len(cells) < 2 or not all(_MARKDOWN_TABLE_DELIMITER_CELL_RE.fullmatch(cell) for cell in cells):
            continue
        header_index = delimiter_index - 1
        if header_index < 0 or "|" not in lines[header_index]:
            continue
        row_numbers.update({header_index + 1, delimiter_index + 1})
        for body_index in range(delimiter_index + 1, len(lines)):
            if not lines[body_index].strip() or "|" not in lines[body_index]:
                break
            row_numbers.add(body_index + 1)
    return row_numbers


def _markdown_column_width(text: str) -> int:
    """Return the visual width of Markdown source using four-column tab stops."""
    column = 0
    for character in text:
        if character == "\t":
            column += _MARKDOWN_TAB_STOP - (column % _MARKDOWN_TAB_STOP)
        else:
            column += 1
    return column


def _markdown_leading_indent(line: str) -> int:
    leading_whitespace = line[: len(line) - len(line.lstrip(" \t"))]
    return _markdown_column_width(leading_whitespace)


def _is_markdown_blockquote_continuation(previous: str, current: str) -> bool:
    """Return whether a quoted source line continues an incomplete statement."""
    dependency_then_bare_bypass = _DEPENDENCY_INSTALL_TERMS_RE.search(previous) and _is_bare_confirmation_bypass(
        current
    )
    bare_bypass_then_dependency = _is_bare_confirmation_bypass(previous) and _DEPENDENCY_INSTALL_TERMS_RE.search(
        current
    )
    if dependency_then_bare_bypass or bare_bypass_then_dependency:
        return True
    if _MARKDOWN_SENTENCE_END_RE.search(previous):
        return False
    first_alpha = re.search(r"[A-Za-z]", current)
    return bool(
        _MARKDOWN_BLOCKQUOTE_CONTINUATION_END_RE.search(previous) or (first_alpha and first_alpha.group(0).islower())
    )


def _is_markdown_fenced_continuation(previous: str, current: str) -> bool:
    """Return whether a fenced literal line wraps the same incomplete statement.

    A fence preserves hard line breaks, so its lines are separate statements
    unless one genuinely wraps. A preceding sentence ending is an unambiguous
    boundary. So is a current line that independently expresses one of the
    policy bypasses this lint recognizes, or a new dependency action following
    such a bypass line.

    Without the second test the outcome turned on punctuation the author never
    wrote: "Install packages" followed by "Never ask for approval." became a
    synthetic bypass. Checking the policy grammar rather than capitalization
    preserves genuine wrapped fragments such as "Preceded by a skill-issued
    prompt or approval request."
    """
    if _MARKDOWN_SENTENCE_END_RE.search(previous):
        return False
    if _is_bare_confirmation_bypass(current) or _has_dependency_review_bypass(current):
        return False
    if _DEPENDENCY_ACTION_AT_START_RE.search(current) and (
        _is_bare_confirmation_bypass(previous) or _has_dependency_review_bypass(previous)
    ):
        return False
    return _is_markdown_blockquote_continuation(previous, current)


def _has_dependency_policy_bypass(text: str, patterns: Iterable[re.Pattern]) -> bool:
    """Return whether dependency guidance contains a semantically linked bypass."""
    return any(pattern.search(text) for pattern in patterns)


def _clause_bounds_at(statement: str, index: int) -> tuple[int, int]:
    """Return bounds of the contrast-or-sequence-delimited clause containing ``index``.

    Used to scope a negation/prohibition check to the specific clause that
    contains a matched "without X" phrase, so an unrelated negation elsewhere in
    the sentence (a different clause) cannot excuse it, while a negation that
    genuinely governs that same clause still can.
    """
    for start, end in _iter_clause_spans(statement):
        if start <= index < end or index < start:
            return start, end
    return 0, len(statement)


def _iter_clause_spans(statement: str) -> list[tuple[int, int]]:
    """Return the contrast-or-sequence-delimited clause spans of ``statement``.

    A coordinated dependency-action series is deliberately not split, so
    "never download, install, or use packages" stays one clause and its leading
    negation keeps governing every item.
    """
    separators = [
        separator
        for separator in _WITHOUT_CLAUSE_BOUNDARY_RE.finditer(statement)
        if not _is_dependency_action_series_boundary(statement, separator)
    ]
    spans = []
    start = 0
    for separator in separators:
        spans.append((start, separator.start()))
        start = separator.end()
    spans.append((start, len(statement)))
    return spans


def _is_dependency_action_series_boundary(statement: str, separator: re.Match) -> bool:
    """Return whether ``, and/or`` joins verbs sharing one negation.

    Coordinators normally start a new clause, but the final coordinator in a
    list such as "Do not download, install, or use packages" only joins verbs.
    Keeping that list together lets the leading negation govern every action.

    List items may carry their own objects, as in "do not download packages,
    install dependencies, or use packages". Test whether the preceding list item
    *starts* with an action rather than ends with one, or such a list splits at
    its final coordinator and the leading negation is lost.
    """
    if not separator.group("coordinator"):
        return False
    preceding = statement[: separator.start()]
    preceding_item = preceding.rsplit(",", 1)[-1]
    return bool(
        (_DEPENDENCY_ACTION_AT_END_RE.search(preceding) or _DEPENDENCY_ACTION_AT_START_RE.search(preceding_item))
        and _DEPENDENCY_ACTION_AT_START_RE.search(statement[separator.end() :])
    )


def _negation_reaches_without_clause(gap: str) -> bool:
    """Return whether a negated verb still governs the action before ``without``.

    ``gap`` is the text between the negated verb and the "without X" phrase. The
    negation still governs in exactly two shapes: the gap is the negated verb's
    own object ("never install project dependencies without X"), or it is a
    coordinated series under the same negation ("never download, install, or
    execute dependencies without X").

    Anything else fails closed. A coordinator in the gap can introduce a second
    action whose verb this module does not recognize -- "never install packages,
    add packages without X" -- and an unrecognized verb must not be mistaken for
    more of the negated verb's object.
    """
    independent_final_action = re.search(
        rf",\s*{_DEPENDENCY_ACTION_PATTERN}\b[^,]*\b{_DEPENDENCY_NOUN_PATTERN}\b\s*$", gap, re.IGNORECASE
    ) and not re.search(r",\s*(?:and|or)\b", gap, re.IGNORECASE)
    if _DEPENDENCY_ACTION_SERIES_GAP_RE.fullmatch(gap) is not None and not independent_final_action:
        return True
    return not _CLAUSE_COORDINATOR_RE.search(gap) and not _DEPENDENCY_ACTION_RE.search(gap)


def _is_read_only_dependency_action(statement: str, match: re.Match) -> bool:
    """Return whether the "without confirmation" phrase governs a read-only action.

    Reading package metadata changes nothing, so it needs no install
    confirmation. Finding a read-only verb somewhere is not enough -- it must be
    the verb the "without" phrase modifies. The exemption is therefore granted
    only to two whole sentence shapes, each of which must consume its side of the
    clause entirely:

        <read-only verb> <object> without <confirmation>
        without <confirmation>, <read-only verb> <object>

    An object word may be neither a coordinator nor a recognized mutating verb,
    so a second action cannot hide inside the phrase: "inspect the package index
    and add dependencies without confirmation" fails to match and stays flagged.
    """
    clause_start, clause_end = _clause_bounds_at(statement, match.start("without_clause"))
    leading = statement[clause_start : match.start("without_clause")].strip(" \t,;")
    trailing = statement[match.end("without_clause") : clause_end].strip(" \t,;.!?")

    if _is_read_only_phrase(leading):
        # Nothing may act after the "without" phrase either.
        return not _CLAUSE_COORDINATOR_RE.search(trailing) and not _DEPENDENCY_ACTION_RE.search(trailing)
    if not leading:
        return _is_read_only_phrase(trailing) and not _DEPENDENCY_ACTION_RE.search(statement[clause_end:])
    return False


def _is_read_only_phrase(phrase: str) -> bool:
    """Return whether ``phrase`` is a whole read-only action phrase.

    "check out packages" is dependency acquisition rather than a read, so the
    phrasal form is rejected even though its verb is otherwise read-only.
    """
    match = _READ_ONLY_ACTION_PHRASE_RE.fullmatch(phrase)
    if not match:
        return False
    tail = phrase[match.end("verb") :]
    return _CHECK_OUT_ACQUISITION_RE.match(phrase) is None and not _tail_introduces_action_gerund(tail)


def _is_read_only_passive_phrase(phrase: str) -> bool:
    """Return whether ``phrase`` is a whole passive read-only action phrase."""
    match = _READ_ONLY_PASSIVE_PHRASE_RE.fullmatch(phrase)
    if not match:
        return False
    tail = phrase[match.end("verb") :]
    return _CHECK_OUT_ACQUISITION_RE.search(phrase) is None and not _tail_introduces_action_gerund(tail)


def _tail_introduces_action_gerund(tail: str) -> bool:
    """Return whether a read-only phrase tail introduces an unknown gerund action."""
    after_preposition = False
    expected_agent_nouns = None
    for word_match in _POLICY_WORD_RE.finditer(tail):
        word = word_match.group(0)
        normalized = word.lower()
        if expected_agent_nouns is not None:
            if normalized not in expected_agent_nouns:
                return True
            expected_agent_nouns = None
            continue
        if normalized in _ACTION_INTRODUCING_PREPOSITIONS:
            after_preposition = True
            continue
        if not after_preposition or not normalized.endswith("ing"):
            continue
        if re.fullmatch(_READ_ONLY_DEPENDENCY_VERB_PATTERN, word, re.IGNORECASE):
            continue
        if normalized in _NON_ACTION_GERUND_AGENT_NOUNS:
            expected_agent_nouns = _NON_ACTION_GERUND_AGENT_NOUNS[normalized]
            continue
        return True
    return expected_agent_nouns is not None


def _is_negated_without_clause(statement: str, match: re.Match) -> bool:
    """Return whether the matched "without X" action is negated into a safe mandate.

    "Install packages without confirmation" is a bypass, but "Never install
    packages without confirmation" and "Installing packages without confirmation
    is prohibited" both require confirmation. A negation elsewhere in the same
    grammatical clause does not make the install safe: in "Do not log secrets
    while installing without confirmation", it governs logging, not installing.
    """
    without_start = match.start("without_clause")
    without_end = match.end("without_clause")
    clause_start, clause_end = _clause_bounds_at(statement, without_start)
    clause = statement[clause_start:clause_end]
    relative_start = without_start - clause_start
    relative_end = without_end - clause_start

    # A trailing prohibition is safe only when it directly predicates the
    # matched "without X" construction. Merely mentioning a prohibited index or
    # source elsewhere in the clause must not excuse an affirmative install.
    if _WITHOUT_CLAUSE_PROHIBITION_TAIL_RE.match(clause[relative_end:]):
        return True

    # Verb ellipsis: "install dependencies, but never without user confirmation"
    # drops the repeated verb, leaving the negation directly against the "without"
    # phrase. Nothing else stands in the clause, so the negation governs it.
    if _ELLIPTICAL_NEGATION_RE.fullmatch(clause[:relative_start]):
        return True

    for pattern in (_NEGATED_DEPENDENCY_ACTION_RE, _PASSIVE_NEGATED_DEPENDENCY_ACTION_RE):
        for negated_action in pattern.finditer(clause):
            if negated_action.end("action") <= relative_start:
                # The negated action must be the action associated with the
                # without-clause. Measure the gap from the negated verb itself,
                # not the end of the whole match: the match runs on to its
                # dependency noun, which can sit past an intervening affirmative
                # verb, as in "do not use unknown indexes, install packages
                # without confirmation".
                if _negation_reaches_without_clause(clause[negated_action.end("action") : relative_start]):
                    return True
            elif negated_action.start() >= relative_end:
                # Supports "Without confirmation, never install packages" but
                # not "Install without confirmation while never using ...".
                if not _DEPENDENCY_ACTION_RE.search(clause[:relative_start]):
                    return True
            else:
                return True
    return False


def _iter_dependency_without_matches(statement: str, pattern: re.Pattern) -> Iterable[re.Match]:
    """Yield every ``without`` occurrence linked to nearby dependency context."""
    for match in pattern.finditer(statement):
        context_start = max(0, match.start("without_clause") - 160)
        context_end = min(len(statement), match.end("without_clause") + 160)
        if _DEPENDENCY_INSTALL_TERMS_RE.search(statement[context_start:context_end]):
            yield match


def _has_dependency_confirmation_without_bypass(statements: list[str]) -> bool:
    for statement in statements:
        for match in _iter_dependency_without_matches(statement, _DEPENDENCY_CONFIRMATION_WITHOUT_RE):
            if _is_read_only_dependency_action(statement, match):
                continue
            if not _is_negated_without_clause(statement, match):
                return True
    return False


def _has_dependency_review_bypass(text: str) -> bool:
    """Distinguish a review bypass from a negated "without audit" mandate."""
    for index, pattern in enumerate(_DEPENDENCY_REVIEW_BYPASS_RES):
        for match in pattern.finditer(text):
            if index == 1 and re.search(
                r"\b(?:so|therefore)\b[^.!?;]{0,40}\b(?:audit|review|vet|classify|flag)\w*\b",
                text[match.end() :],
                re.IGNORECASE,
            ):
                continue
            if index < 2 or not _policy_action_is_negated(text, match.start()):
                return True
    statements = _split_policy_statements(text)
    for statement in statements:
        for match in _iter_dependency_without_matches(statement, _DEPENDENCY_REVIEW_WITHOUT_RE):
            if not _is_negated_without_clause(statement, match):
                return True
    return False


def _is_bare_confirmation_bypass(text: str) -> bool:
    return bool(_BARE_CONFIRMATION_BYPASS_RE.fullmatch(text) or _BARE_CONFIRMATION_DENIAL_RE.fullmatch(text))


def _has_actionable_dependency_context(statements: list[str], excluded_index: int) -> bool:
    """Return whether another statement permits a dependency-related action.

    Judged per clause rather than per statement: a statement can both forbid one
    action and permit another, as in "never download unknown packages, but
    install dependencies". Discarding the whole statement because it contains a
    negation would let the permitted install go unnoticed, so a neighbouring bare
    "never ask for confirmation" would not be recognized as a bypass.

    A purely read-only clause does not permit a mutation, so it is not actionable
    context either -- in the passive voice as much as the active.
    """
    for index, statement in enumerate(statements):
        if index == excluded_index or not _DEPENDENCY_INSTALL_TERMS_RE.search(statement):
            continue
        for start, end in _iter_clause_spans(statement):
            clause = statement[start:end]
            if not _DEPENDENCY_INSTALL_TERMS_RE.search(clause):
                continue
            if _is_read_only_clause(clause):
                continue
            actions = list(_DEPENDENCY_ACTION_RE.finditer(clause))
            if not actions and _PROHIBITED_DEPENDENCY_CONTEXT_RE.search(clause):
                continue
            if not actions or _has_uncovered_dependency_action(clause, actions):
                return True
    return False


def _has_uncovered_dependency_action(clause: str, actions: list[re.Match]) -> bool:
    """Return whether any recognized action is not governed by a prohibition.

    A negated or prohibited action covers only its own verb. A leading active or
    passive negation may additionally govern a canonical coordinated verb list,
    but not a new ``, and/or`` clause or a later subject-plus-predicate action.
    """
    covered_spans = set()
    negated_matches = list(_NEGATED_DEPENDENCY_ACTION_RE.finditer(clause)) + list(
        _PASSIVE_NEGATED_DEPENDENCY_ACTION_RE.finditer(clause)
    )
    for match in negated_matches:
        covered_spans.add(match.span("action"))

    for match in _PROHIBITED_DEPENDENCY_ACTION_RE.finditer(clause):
        covered_spans.update(action.span() for action in actions if match.start() <= action.start() < match.end())

    for match in negated_matches:
        raw_tail = clause[match.end("action") :]
        if re.match(r"^\s*,\s*(?:and|or)\b", raw_tail, re.IGNORECASE):
            continue
        tail = raw_tail.strip(" \t.!?;")
        if tail and _DEPENDENCY_ACTION_SERIES_GAP_RE.fullmatch(tail):
            covered_spans.update(action.span() for action in actions if action.start() > match.end("action"))

    return any(action.span() not in covered_spans for action in actions)


def _is_read_only_clause(clause: str) -> bool:
    """Return whether a clause only reads dependency state.

    Covers the active phrase ("inspect package metadata") and its passive
    equivalent ("package metadata must be inspected"). A recognized mutating
    verb disqualifies the passive form, so "packages must be installed" stays
    actionable.
    """
    text = clause.strip(" \t,;.!?")
    if _is_read_only_phrase(text):
        return True
    if _DEPENDENCY_ACTION_RE.search(text):
        return False
    return _is_read_only_passive_phrase(text)


def _has_nearby_audit_then_confirm(statements: list[str], index: int) -> bool:
    """Return whether an audit-first/post-audit-confirmation pair covers ``index``.

    The flagged statement itself must be part of the audit-then-confirm sequence --
    either alone, or paired with its immediate left or right neighbor -- not merely
    within a wider window. A three-statement window would let an unrelated bypass
    statement sit between a real audit-first statement and its real post-audit
    confirmation and be exempted by proximity alone, even though it is not actually
    part of that sequence.
    """
    flagged_statement = statements[index]
    if not (
        _DEPENDENCY_AUDIT_FIRST_RE.search(flagged_statement)
        or _has_nonnegated_post_audit_confirmation(flagged_statement)
    ):
        return False

    candidate_spans = [(index, index)]
    if index > 0:
        candidate_spans.append((index - 1, index))
    if index + 1 < len(statements):
        candidate_spans.append((index, index + 1))
    for start, end in candidate_spans:
        window_text = " ".join(statements[start : end + 1])
        if _DEPENDENCY_AUDIT_FIRST_RE.search(window_text) and _has_nonnegated_post_audit_confirmation(window_text):
            return True
    return False


def _has_dependency_confirmation_bypass(text: str) -> bool:
    """Distinguish a confirmation bypass from an explicit audit-then-confirm sequence."""
    if re.fullmatch(
        r"no\s+(?:user\s+)?(?:approval|confirmation|consent|permission)\s+(?:is\s+)?(?:required|needed)"
        r"\s+for\s+packages?\s+(?:that\s+)?(?:the\s+)?user\s+(?:already|previously)\s+"
        r"(?:approved|confirmed|authorized)(?:\s+in\s+the\s+install\s+plan)?[.!?;]?",
        text.strip(),
        re.IGNORECASE,
    ):
        return False
    statements = _split_policy_statements(text)
    for pattern_index, pattern in enumerate(_DEPENDENCY_CONFIRMATION_BYPASS_RES):
        for match in pattern.finditer(text):
            if pattern_index == 1:
                statement_index = next(
                    (index for index, statement in enumerate(statements) if match.group(0) in statement), None
                )
                if statement_index is not None and _has_nearby_audit_then_confirm(statements, statement_index):
                    continue
            return True
    if _has_dependency_confirmation_without_bypass(statements):
        return True
    for index, statement in enumerate(statements):
        if _BARE_CONFIRMATION_DENIAL_RE.fullmatch(statement) and _has_actionable_dependency_context(statements, index):
            return True
    for index, statement in enumerate(statements):
        bare_confirmation_suppression = _BARE_CONFIRMATION_BYPASS_RE.fullmatch(
            statement
        ) and _has_actionable_dependency_context(statements, index)
        embedded_bare_suppression = _BARE_CONFIRMATION_BYPASS_CLAUSE_RE.search(
            statement
        ) and _DEPENDENCY_CONFIRMATION_REQUEST_SUPPRESSION_RE.search(statement)
        request_suppression = _DEPENDENCY_CONFIRMATION_REQUEST_SUPPRESSION_RE.search(statement)
        if embedded_bare_suppression:
            return True
        if not bare_confirmation_suppression and not request_suppression:
            continue
        if not _has_nearby_audit_then_confirm(statements, index):
            return True
    return False


def _split_policy_statements(text: str) -> list[str]:
    """Split policy prose without treating common abbreviation dots as stops."""
    protected = re.sub(
        r"\b(?:e\.g\.|i\.e\.)", lambda match: match.group(0).replace(".", "\u2024"), text, flags=re.IGNORECASE
    )
    statements = [statement.strip() for statement in re.split(r"(?<=[.!?;])\s+", protected) if statement.strip()]
    return [statement.replace("\u2024", ".") for statement in statements]


def _policy_action_is_negated(text: str, action_start: int) -> bool:
    """Return whether clause-local syntax negates a matched policy action."""
    clause_start, _ = _clause_bounds_at(text, action_start)
    prefix = text[clause_start:action_start]
    return bool(
        re.search(
            r"(?:\bunder\s+no\s+circumstances\b|\bnever(?:\s*,?\s*ever)?\b|\bavoid(?:s|ed|ing)?\b|"
            r"\b(?:do|does|did|must|should|shall|may|can|could|will|would)\s+not\b|\bcannot\b|"
            r"\b(?:don|doesn|didn|mustn|shouldn|shan|mayn|can|couldn|won|wouldn)['’]t\b)"
            r"[^.!?;|]{0,48}$",
            prefix,
            re.IGNORECASE,
        )
    )


def _has_nonnegated_post_audit_confirmation(text: str) -> bool:
    """Return whether text positively requires confirmation after the audit."""
    for pattern in _DEPENDENCY_POST_AUDIT_CONFIRMATION_RES:
        for match in pattern.finditer(text):
            matched = match.group(0)
            if not re.search(
                r"\b(?:obtain|request|receive|require|wait\s+for)\b[^.!?;]{0,80}"
                r"\b(?:no|not|never)\b[^.!?;]{0,40}\b(?:approval|confirmation|consent|permission)\b",
                matched,
                re.IGNORECASE,
            ):
                return True
    return False


def _eval_mentions_file_editing(item: dict[str, Any]) -> bool:
    text = _eval_text(item).lower()
    patterns = (
        r"\b(?:edit|modify|update|rewrite|write|create|generate|export)s?\s+(?:a\s+|an\s+|the\s+)?file\b",
        r"\b(?:edit|modify|update|rewrite|write|create|generate|export)s?\s+(?:source|code|artifact)s?\b",
        r"\b(?:file|artifact)s?\s+(?:is|are|must be|should be)?\s*(?:created|generated|written|exported|modified)\b",
        r"\boutput\s+(?:file|artifact|directory)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _has_files(path: Path) -> bool:
    return path.is_dir() and any(True for _child in _iter_files_no_follow(path))


def _walk_no_follow(
    root: Path, excluded_dir_names: Iterable[str] = frozenset()
) -> Iterable[tuple[Path, list[str], list[str]]]:
    """Deterministic os.walk that never follows symlinks and prunes excluded dirs.

    Yields ``(current_dir_path, dir_names, file_names)`` with both name lists
    sorted; ``dir_names`` is the live pruned list, so callers may remove entries
    to stop descent into those directories.
    """
    excluded_dir_names = set(excluded_dir_names)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current_dir = Path(dirpath)
        dirnames[:] = sorted(
            name for name in dirnames if name not in excluded_dir_names and not (current_dir / name).is_symlink()
        )
        yield current_dir, dirnames, sorted(filenames)


def _iter_files_no_follow(root: Path, *, excluded_dir_names: Iterable[str] = ()) -> Iterable[Path]:
    if root.is_symlink() or not root.is_dir():
        return
    for current_dir, _dir_names, file_names in _walk_no_follow(root, excluded_dir_names):
        for filename in file_names:
            path = current_dir / filename
            # Reject symlinked files too, not just symlinked dirs: a
            # references/x.md -> /etc/passwd symlink is a "file" to is_file()
            # and would otherwise be read from outside the skill tree.
            if path.is_file() and not path.is_symlink():
                yield path


def _has_fixture_notes(evals_dir: Path) -> bool:
    note_paths = (
        evals_dir / "README.md",
        evals_dir / "files" / "README.md",
        evals_dir / "files" / "SOURCE.md",
    )
    return any(path.is_file() for path in note_paths)


def _read_bounded_text(path: Path) -> Optional[str]:
    try:
        return read_regular_text_file(path, max_bytes=MAX_SKILL_TEXT_FILE_BYTES, errors="replace")
    except (OSError, UnicodeError, ValueError):
        return None


def _is_oversized_text_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > MAX_SKILL_TEXT_FILE_BYTES
    except OSError:
        return False


def _has_bounded_size_exception(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(16 * 1024)
    except OSError:
        return False
    return _has_size_exception(prefix.decode("utf-8", errors="replace"))


def _line_for_frontmatter_issue(skill_file: Path, code: str, message: str) -> Optional[int]:
    if code == "skill-frontmatter-field-required":
        match = re.search(r"field '([^']+)'", message)
        if match:
            return _line_for_field(skill_file, match.group(1))
    if code in {
        "skill-name-directory-mismatch",
        "skill-blast-radius-invalid",
        "skill-frontmatter-field-type",
        "skill-frontmatter-field-unsupported",
    }:
        for field in ("name", "blast-radius", "description", "min-flare-version", "category"):
            if field in message:
                return _line_for_field(skill_file, field)
    return 1 if skill_file.is_file() else None


def _line_for_field(skill_file: Path, field: str) -> Optional[int]:
    if not skill_file.is_file():
        return None
    if _is_oversized_text_file(skill_file):
        return 1
    prefix = f"{field}:"
    for line_no, line in enumerate(skill_file.read_text(encoding="utf-8-sig", errors="replace").splitlines(), start=1):
        if line.strip().startswith(prefix):
            return line_no
    return 1


def _has_size_exception(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SIZE_EXCEPTION_MARKERS)


def _skip(context: LintContext, check: str, reason: str) -> None:
    context.skipped_checks.append({"id": check, "reason": reason})


def _finding(lint_id: str, severity: str, path: Path, message: str, hint: str, **kwargs: Any) -> LintFinding:
    """Build a LintFinding, converting the path to the string ``file`` field."""
    return LintFinding(id=lint_id, severity=severity, file=str(path), message=message, hint=hint, **kwargs)
