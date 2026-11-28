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

import json
import os
import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import requests
import wikipedia
from mas.langchain_compat import Document

# Common medical/pharma abbreviations -> Wikipedia-friendly full names
MEDICAL_ABBREVIATIONS: Dict[str, str] = {
    "NPY": "Neuropeptide Y",
    "ADH": "Antidiuretic hormone",
    "ACTH": "Adrenocorticotropic hormone",
    "GABA": "Gamma-Aminobutyric acid",
    "GnRH": "Gonadotropin-releasing hormone",
    "FSH": "Follicle-stimulating hormone",
    "LH": "Luteinizing hormone",
    "TSH": "Thyroid-stimulating hormone",
    "PCT": "Proximal convoluted tubule",
    "DCT": "Distal convoluted tubule",
    "LOH": "Loop of Henle",
    "REM": "Rapid eye movement sleep",
    "EEG": "Electroencephalography",
    "cAMP": "Cyclic adenosine monophosphate",
    "GPCR": "G protein-coupled receptor",
    "ACE": "Angiotensin-converting enzyme",
    "COX": "Cyclooxygenase",
    "CYP": "Cytochrome P450",
    "MT1": "Melatonin receptor 1",
    "MT2": "Melatonin receptor 2",
    "SNRI": "Serotonin-norepinephrine reuptake inhibitor",
    "SSRI": "Selective serotonin reuptake inhibitor",
}


def normalize_search_query(query: str, max_words: int = 6) -> str:
    """Extract a concise, Wikipedia-friendly search term from a long or messy query."""
    if not query or not isinstance(query, str):
        return query
    raw = query.strip()
    # Expand known abbreviations (case-insensitive match, prefer whole-word)
    for abbr, full in MEDICAL_ABBREVIATIONS.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        if re.search(pattern, raw, re.IGNORECASE):
            raw = re.sub(pattern, full, raw, flags=re.IGNORECASE)
    # If short enough, return as-is (possibly after abbreviation expansion)
    words = raw.split()
    if len(words) <= max_words and len(raw) <= 80:
        return " ".join(words)
    # Long query: take first segment (before comma/semicolon/period) or first max_words words
    segment = re.split(r"[,;.]", raw, maxsplit=1)[0].strip()
    segment_words = segment.split()
    if len(segment_words) <= max_words:
        return " ".join(segment_words)
    # Take first max_words words; prefer title-case phrases (likely entity names)
    return " ".join(segment_words[:max_words])


class LangChainWiki:

    def __init__(self, cache_path: Optional[str] = None) -> None:
        self.document: Optional[Document] = None
        self.lookup_str = ""
        self.lookup_index = 0
        self.cache_path = cache_path or os.environ.get("NV_WIKI_SEARCH_CACHE", "")
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache(self.cache_path)
        self._last_errors: List[str] = []
        self._last_disambiguation_options: List[str] = []

    @staticmethod
    def _load_cache(cache_path: str) -> Dict[str, Dict[str, Any]]:
        if not cache_path or cache_path.lower() in {"0", "false", "off", "none"}:
            return {}
        try:
            if not os.path.exists(cache_path):
                return {}
            with open(cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_cache(self) -> None:
        if not self.cache_path or self.cache_path.lower() in {"0", "false", "off", "none"}:
            return
        try:
            cache_dir = os.path.dirname(self.cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            existing = self._load_cache(self.cache_path)
            if existing:
                existing.update(self._cache)
                self._cache = existing
            tmp_path = f"{self.cache_path}.{os.getpid()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self._cache, fh, ensure_ascii=False)
            os.replace(tmp_path, self.cache_path)
        except Exception:
            return

    @staticmethod
    def _cache_key(search: str, context: str = "") -> str:
        query = normalize_search_query(search or "")
        compact_context = " ".join(str(context or "").lower().split())
        return json.dumps(
            {"query": query.lower(), "context": compact_context[:240]},
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _dedupe(items: List[str]) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for item in items:
            value = str(item or "").strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result

    @staticmethod
    def _claim_types(context: str) -> set[str]:
        text = str(context or "").lower()
        found: set[str] = set()
        if re.search(r"\b(film|movie|cinema|directed|starring)\b", text):
            found.add("film")
        if re.search(r"\b(tv|television|series|show|season|episode|cancelled|canceled|renewed)\b", text):
            found.add("television")
        if re.search(r"\b(song|single|album|band|singer|rapper|musician|record)\b", text):
            found.add("music")
        return found

    def _contextual_queries(self, query: str, context: str = "") -> List[str]:
        query = normalize_search_query(query or "")
        claim_types = self._claim_types(context)
        variants = [query]
        lowered = query.lower()

        if "film" in claim_types and not re.search(r"\b(film|movie)\b", lowered):
            variants.extend([f"{query} film", f"{query} (film)"])
        if "television" in claim_types and not re.search(r"\b(tv|television|series|show|season)\b", lowered):
            variants.extend([f"{query} TV series", f"{query} television series", f"{query} (TV series)"])
        if "music" in claim_types and not re.search(
            r"\b(song|single|album|band|singer|rapper|musician|record)\b", lowered
        ):
            variants.extend([f"{query} song", f"{query} album", f"{query} band"])

        return self._dedupe(variants)

    @staticmethod
    def _doc_is_context_match(doc: Document, context: str = "") -> bool:
        claim_types = LangChainWiki._claim_types(context)
        if not claim_types:
            return True
        text = f"{doc.metadata.get('title', '')} {doc.page_content[:1200]}".lower()
        if "film" in claim_types and re.search(r"\b(film|movie|directed by|starring)\b", text):
            return True
        if "television" in claim_types and re.search(
            r"\b(tv|television|series|show|season|episode|renewed|cancelled|canceled)\b", text
        ):
            return True
        if "music" in claim_types and re.search(r"\b(song|single|album|band|singer|rapper|musician|record)\b", text):
            return True
        return False

    def _cache_doc(self, key: str, doc: Document, *, search: str, query: str, stage: str, similar: List[str]) -> None:
        self._cache[key] = {
            "status": "hit",
            "search": search,
            "query": query,
            "stage": stage,
            "title": doc.metadata.get("title", ""),
            "url": doc.metadata.get("page", ""),
            "content": doc.page_content,
            "similar": similar[:8],
            "errors": list(self._last_errors[-8:]),
        }
        self._save_cache()

    def _cache_failure(self, key: str, *, search: str, query: str, similar: List[str]) -> str:
        similar_str = ", ".join(similar[:8]) if similar else "none"
        error_str = "; ".join(self._last_errors[-3:])
        observation = f"Could not find [{search}]. Similar: [{similar_str}]"
        if error_str:
            observation = f"{observation} SearchErrors: [{error_str}]"
        if error_str:
            return observation
        self._cache[key] = {
            "status": "miss",
            "search": search,
            "query": query,
            "observation": observation,
            "similar": similar[:8],
            "errors": list(self._last_errors[-8:]),
        }
        self._save_cache()
        return observation

    def _apply_cache_entry(self, entry: Dict[str, Any]) -> Optional[str]:
        if not isinstance(entry, dict):
            return None
        if entry.get("status") == "hit":
            self.document = Document(
                page_content=str(entry.get("content", "") or ""),
                metadata={"page": str(entry.get("url", "") or ""), "title": str(entry.get("title", "") or "")},
            )
            self.lookup_str = ""
            self.lookup_index = 0
            return self._sumary
        if entry.get("status") == "miss":
            errors = " ".join(str(x) for x in (entry.get("errors") or []))
            observation = str(entry.get("observation") or "")
            if "JSONDecodeError" in errors or "JSONDecodeError" in observation:
                return None
            self.document = None
            return observation or "Could not find page."
        return None

    def _apply_query_cache_hit(self, query: str) -> Optional[str]:
        query_norm = normalize_search_query(query or "").strip().lower()
        if not query_norm:
            return None
        for entry in self._cache.values():
            if not isinstance(entry, dict) or entry.get("status") != "hit":
                continue
            cached_query = normalize_search_query(str(entry.get("query") or "")).strip().lower()
            if cached_query != query_norm:
                continue
            return self._apply_cache_entry(entry)
        return None

    def _try_page(self, title: str) -> Optional[Document]:
        """Try to load a Wikipedia page by title. Returns Document or None."""
        self._last_disambiguation_options = []
        try:
            page = wikipedia.page(title, auto_suggest=False)
            return Document(
                page_content=page.content,
                metadata={"page": page.url, "title": getattr(page, "title", title)},
            )
        except wikipedia.DisambiguationError as exc:
            self._last_disambiguation_options = list(getattr(exc, "options", []) or [])
            self._last_errors.append(f"DisambiguationError({title})")
            return None
        except wikipedia.PageError:
            self._last_errors.append(f"PageError({title})")
            return None
        except Exception as exc:
            self._last_errors.append(f"{type(exc).__name__}({title})")
            return None

    def _similar_titles(self, query: str) -> List[str]:
        try:
            return list(wikipedia.search(query) or [])
        except Exception as exc:
            self._last_errors.append(f"{type(exc).__name__}:wikipedia.search({query})")
            return []

    def search(self, search: str, context: str = "") -> Union[str, Document]:
        query = normalize_search_query(search)
        cache_key = self._cache_key(search, context)
        cached = self._apply_cache_entry(self._cache.get(cache_key, {}))
        if cached is not None:
            return cached
        if not str(context or "").strip():
            cached = self._apply_query_cache_hit(query)
            if cached is not None:
                return cached

        self._last_errors = []
        queries = self._contextual_queries(query, context)
        fallback_doc: Optional[Document] = None
        fallback_query = query
        similar: List[str] = []
        candidate_titles: List[str] = []

        # 1) Try exact/contextual queries. If an exact page is too broad for the
        # claim type, keep it as a fallback and try typed variants before returning.
        for candidate_query in queries:
            doc = self._try_page(candidate_query)
            candidate_titles.extend(self._last_disambiguation_options)
            if doc is None:
                continue
            if self._doc_is_context_match(doc, context):
                self.document = doc
                self.lookup_str = ""
                self.lookup_index = 0
                self._cache_doc(
                    cache_key, doc, search=search, query=candidate_query, stage="exact_or_context", similar=similar
                )
                return self._sumary
            if fallback_doc is None:
                fallback_doc = doc
                fallback_query = candidate_query

        # 2) Get similar/disambiguation titles and retry ranked candidates.
        for candidate_query in queries:
            similar.extend(self._similar_titles(candidate_query))
        candidate_titles.extend(similar)
        ranked_titles = sorted(
            self._dedupe(candidate_titles),
            key=lambda title: int(self._title_matches_claim_type(title, context)),
            reverse=True,
        )
        for title in ranked_titles[:10]:
            if not title or title.lower() in {q.lower() for q in queries}:
                continue
            doc = self._try_page(title)
            if doc is None:
                continue
            if self._doc_is_context_match(doc, context) or fallback_doc is None:
                self.document = doc
                self.lookup_str = ""
                self.lookup_index = 0
                self._cache_doc(
                    cache_key, doc, search=search, query=title, stage="similar_or_disambiguation", similar=similar
                )
                return self._sumary

        # 3) If no typed candidate works, return the exact broad page rather than
        # manufacturing a miss from a failed secondary query.
        if fallback_doc is not None:
            self.document = fallback_doc
            self.lookup_str = ""
            self.lookup_index = 0
            self._cache_doc(
                cache_key, fallback_doc, search=search, query=fallback_query, stage="broad_fallback", similar=similar
            )
            return self._sumary

        # 4) All failed: return message with similar list and visible error type.
        self.document = None
        return self._cache_failure(cache_key, search=search, query=query, similar=self._dedupe(similar))

    @staticmethod
    def _title_matches_claim_type(title: str, context: str = "") -> bool:
        claim_types = LangChainWiki._claim_types(context)
        lowered = str(title or "").lower()
        if "film" in claim_types and re.search(r"\b(film|movie)\b", lowered):
            return True
        if "television" in claim_types and re.search(r"\b(tv|television|series|show|season)\b", lowered):
            return True
        if "music" in claim_types and re.search(r"\b(song|single|album|band|record)\b", lowered):
            return True
        return False

    def lookup(self, term: str):

        if self.document is None:
            raise ValueError("Cannot lookup without a successful search first")
        if term.lower() != self.lookup_str:
            self.lookup_str = term.lower()
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self._paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_index >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"

    @property
    def _sumary(self) -> str:
        return self._paragraphs[0]

    @property
    def _paragraphs(self) -> list[str]:
        if self.document is None:
            raise ValueError("Cannot get paragraphs without a document")
        return self.document.page_content.split("\n\n")


# ------------------------------ SearXNG + Crawl4AI structured search ------------------------------

SEARXNG_SEARCH_URL_ENV = "SEARXNG_SEARCH_URL"
SEARXNG_DEFAULT_URL = "http://localhost:8080/search"

TAVILY_API_KEY_ENV = "TAVILY_API_KEY"
TAVILY_DEFAULT_URL = "https://api.tavily.com/search"


@dataclass
class StructuredSearchParams:
    """
    Structured search schema agreed between agent and search backend.

    The LLM is expected to output a JSON object with fields:
    - keywords: core search keywords (entity / event names)
    - constraints: negative or restrictive conditions (may be natural language)
    - time_range: time range description (e.g. past_24_hours, 2025-2026)
    - source_type: preferred source type (e.g. academic, official news, social media)
    - reformulated_queries: at least 1-3 reformulated search queries
    """

    keywords: str
    constraints: str = ""
    time_range: str = ""
    source_type: str = ""
    reformulated_queries: List[str] = field(default_factory=list)


@dataclass
class TavilySearchClient:
    """
    Unified Tavily search client for agents.

    Capabilities:
    - Accept StructuredSearchParams (including reformulated_queries)
    - Call Tavily /search for multiple queries
    - Simple ranking (keyword hits + Tavily's own score)
    - Return I_NEED_MORE_INFO when results are too few / uninformative
    """

    api_key: Optional[str] = None
    base_url: str = TAVILY_DEFAULT_URL
    max_results: int = 10

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get(TAVILY_API_KEY_ENV)
        if not self.api_key:
            raise RuntimeError(
                f"Tavily API key not found. " f"Set environment variable {TAVILY_API_KEY_ENV} to your Tavily key."
            )

    def _request(self, query: str, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Call Tavily /search HTTP endpoint.
        Reference: https://docs.tavily.com/documentation/api-reference/endpoint/search
        """
        payload: Dict[str, Any] = {
            "query": query,
            "search_depth": "advanced",
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        if topic:
            payload["topic"] = topic

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _simple_rank_results(
        results: List[Dict[str, Any]],
        keywords: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Non-learning ranker:
        - first check how many content words from keywords appear in text
        - then use Tavily's own score (if present) as a secondary signal
        """
        if not results:
            return []

        terms = {t.lower() for t in re.split(r"\W+", keywords) if len(t) > 2}

        def score_item(item: Dict[str, Any]) -> tuple[int, float]:
            text = f"{item.get('title', '')} {item.get('content', '')}".lower()
            hit = sum(1 for t in terms if t in text)
            base_score = float(item.get("score", 0.0) or 0.0)
            return hit, base_score

        ranked = sorted(results, key=score_item, reverse=True)
        return ranked[:top_k]

    def structured_search(
        self,
        params: StructuredSearchParams,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Unified entry point:
        - Accept StructuredSearchParams (including reformulated_queries)
        - Perform multi-query search
        - Merge results and apply simple ranking
        - Return a dict in the form:
          {
            "status": "OK" | "I_NEED_MORE_INFO",
            "primary_query": ...,
            "reformulated_queries": [...],
            "used_queries": [...],
            "top_results": [
              {"title": ..., "url": ..., "snippet": ...},
              ...
            ]
          }
        """
        # 1) Normalize primary keywords, reusing normalize_search_query
        primary = normalize_search_query(params.keywords, max_words=8)

        # 2) Collect and deduplicate all candidate queries
        candidates: List[str] = [primary] if primary else []
        for q in params.reformulated_queries:
            if q and q not in candidates:
                candidates.append(q)

        # If the LLM does not provide reformulated_queries, fall back to keywords only
        if not candidates and params.keywords:
            candidates.append(params.keywords.strip())

        # Concatenate constraints / time_range into the query text so Tavily can use them as strong signals
        suffix_parts = []
        if params.constraints:
            suffix_parts.append(params.constraints)
        if params.time_range:
            suffix_parts.append(params.time_range)
        suffix = " ".join(suffix_parts).strip()

        # source_type -> topic (very rough mapping, only as a weak preference)
        topic = None
        if "news" in (params.source_type or "").lower():
            topic = "news"

        all_results: List[Dict[str, Any]] = []
        used_queries: List[str] = []
        for q in candidates:
            full_query = f"{q} {suffix}".strip() if suffix else q
            if not full_query:
                continue
            try:
                data = self._request(full_query, topic=topic)
            except Exception as exc:
                # Conservative handling: a single failed request should not break the whole run.
                print(f"[TavilySearchClient] request failed for query={full_query!r}: {exc}")
                continue
            # Tavily responses usually contain a "results" field
            results = data.get("results") or []
            if results:
                used_queries.append(full_query)
                all_results.extend(results)

        if not all_results:
            return {
                "status": "I_NEED_MORE_INFO",
                "primary_query": primary,
                "reformulated_queries": params.reformulated_queries,
                "used_queries": used_queries,
                "top_results": [],
                "reason": "No relevant results from Tavily. Please ask the user for more specific information.",
            }

        ranked = self._simple_rank_results(
            all_results,
            keywords=primary or params.keywords,
            top_k=top_k,
        )
        top_results = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content") or item.get("snippet") or "",
            }
            for item in ranked
        ]

        return {
            "status": "OK",
            "primary_query": primary,
            "reformulated_queries": params.reformulated_queries,
            "used_queries": used_queries,
            "top_results": top_results,
        }


@dataclass
class SearxngSearchClient:
    """
    Structured search client based on local SearXNG + Crawl4AI.

    Pipeline:
    1. SearXNG returns candidates for multiple queries (keywords + reformulated_queries).
    2. Pre-filter by SearXNG's own score to take global top 10.
    3. Re-rank these 10 by keyword matching, and return top-3 to the LLM.
    """

    base_url: str = field(default_factory=lambda: os.environ.get(SEARXNG_SEARCH_URL_ENV, SEARXNG_DEFAULT_URL))
    timeout: int = 20

    def _request(self, query: str) -> List[Dict[str, Any]]:
        """
        Call local SearXNG /search endpoint and return the raw JSON result list.
        Make sure SearXNG is deployed locally and JSON API is enabled.
        """
        if not self.base_url:
            raise RuntimeError(
                "SearXNG search URL is not configured. "
                f"Set environment variable {SEARXNG_SEARCH_URL_ENV} or "
                "specify base_url explicitly when creating SearxngSearchClient."
            )
        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "safesearch": 1,
            "categories": "general",
        }
        headers = {
            # Pretend to be a common browser user agent to avoid some instances blocking python-requests
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
            ),
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(self.base_url, params=params, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", []) or []

    @staticmethod
    def _lexical_score(text: str, keywords: str) -> int:
        """
        Simple token matching score: count how many content words from keywords
        appear in the given text.
        """
        if not text or not keywords:
            return 0
        text_l = text.lower()
        terms = {t.lower() for t in re.split(r"\W+", keywords) if len(t) > 2}
        return sum(1 for t in terms if t in text_l)

    def _rerank_topk(
        self,
        results: List[Dict[str, Any]],
        keywords: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Light-weight rerank over the already score-filtered SearXNG results.
        By default return top_k items (upstream truncates to top-10 first).
        """
        if not results:
            return []

        def score_item(item: Dict[str, Any]) -> tuple[int, float]:
            base_score = float(item.get("score", 0.0) or 0.0)
            text = f"{item.get('title', '')} {item.get('content', '')} {item.get('snippet', '')}"
            lex = self._lexical_score(text, keywords)
            # First compare lexical hits, then fall back to the original score
            return lex, base_score

        ranked = sorted(results, key=score_item, reverse=True)
        return ranked[:top_k]

    def structured_search(
        self,
        params: StructuredSearchParams,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Unified entry point:
        - Accept StructuredSearchParams (including reformulated_queries)
        - Query SearXNG, merge results, and take global top-10 by score
        - Rerank within top-10 and return top_k (default 3)
        """
        primary = normalize_search_query(params.keywords, max_words=8)

        # Build candidate queries
        candidates: List[str] = [primary] if primary else []
        for q in params.reformulated_queries:
            if q and q not in candidates:
                candidates.append(q)
        if not candidates and params.keywords:
            candidates.append(params.keywords.strip())

        # Concatenate constraints / time_range directly into the query text for SearXNG
        suffix_parts = []
        if params.constraints:
            suffix_parts.append(params.constraints)
        if params.time_range:
            suffix_parts.append(params.time_range)
        suffix = " ".join(suffix_parts).strip()

        all_results: List[Dict[str, Any]] = []
        used_queries: List[str] = []

        for q in candidates:
            full_query = f"{q} {suffix}".strip() if suffix else q
            if not full_query:
                continue
            try:
                results = self._request(full_query)
            except Exception as exc:
                print(f"[SearxngSearchClient] request failed for query={full_query!r}: {exc}")
                continue
            if results:
                used_queries.append(full_query)
                all_results.extend(results)

        if not all_results:
            return {
                "status": "I_NEED_MORE_INFO",
                "primary_query": primary,
                "reformulated_queries": params.reformulated_queries,
                "used_queries": used_queries,
                "top_results": [],
                "reason": "No relevant results from SearXNG. Please ask the user for more specific information.",
            }

        # First sort by SearXNG's own score and keep global top-10
        def base_sort(item: Dict[str, Any]) -> float:
            return float(item.get("score", 0.0) or 0.0)

        # De-duplicate results by URL
        seen_urls = set()
        dedup_results: List[Dict[str, Any]] = []
        for item in sorted(all_results, key=base_sort, reverse=True):
            url = item.get("url")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            dedup_results.append(item)

        top10 = dedup_results[:10]

        # Rerank within the top-10 and select top_k (default 3) for the LLM
        reranked = self._rerank_topk(
            top10,
            keywords=primary or params.keywords,
            top_k=top_k,
        )

        top_results = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content") or item.get("snippet") or "",
            }
            for item in reranked
        ]

        return {
            "status": "OK",
            "primary_query": primary,
            "reformulated_queries": params.reformulated_queries,
            "used_queries": used_queries,
            "top_results": top_results,
        }


def normalize_answer(s: str):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def match_exactly(answer, key) -> bool:

    n_answer = normalize_answer(answer)
    n_key = normalize_answer(key)
    return n_answer == n_key
