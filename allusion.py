"""
Allusion v2
A privacy-friendly public-web exploration agent.

What it does:
- Searches public web results with DDGS
- Safely fetches and parses readable text
- Skips non-text/binary pages instead of crashing
- Scores "niche" sources
- Separates mainstream signals from niche signals
- Produces a cleaner Markdown intelligence brief

Privacy model:
- No private accounts
- No email/calendar/contact access
- No scraping behind logins
- Only public URLs returned from search
"""

from __future__ import annotations

import json

import argparse
import datetime as dt
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS
from pydantic import BaseModel, Field


def load_word_set(filename: str) -> set[str]:
    path = Path("config") / filename

    if not path.exists():
        print(f"[warning] Missing config file: {path}")
        return set()

    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


STOPWORDS = load_word_set("stopwords.txt")

NICHE_TERMS = load_word_set("niche_terms.txt")

MAINSTREAM_DOMAINS = load_word_set("mainstream_domains.txt")

HIGH_SIGNAL_DOMAINS = load_word_set("high_signal_domains.txt")

BLOCKED_DOMAINS = load_word_set("blocked_domains.txt")

IGNORED_ALLUSION_TERMS = load_word_set("ignored_allusion_terms.txt")


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class SourceDoc(BaseModel):
    title: str
    url: str
    domain: str
    snippet: str = ""
    text: str = ""
    status: str = "ok"
    niche_score: float = 0.0
    mainstream_score: float = 0.0
    niche_reasons: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class Observation(BaseModel):
    title: str
    insight: str
    evidence: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    category: str = "pattern"


class AnalyticalFinding(BaseModel):
    title: str
    finding: str
    implication: str
    evidence: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class AllusionReport(BaseModel):
    topic: str
    generated_at: str
    query: str
    sources: List[SourceDoc]
    themes: List[str]
    niche_signals: List[str]
    mainstream_signals: List[str]
    allusions: List[str]
    observations: List[Observation]
    analytical_findings: List[AnalyticalFinding]
    cautions: List[str]
    markdown: str


class AllusionAgent:
    def __init__(
        self,
        delay_seconds: float = 0.75,
        timeout: int = 15,
        include_social: bool = False,
        verbose: bool = False,
    ):
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self.include_social = include_social
        self.verbose = verbose

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "AllusionResearchAgent/0.2 "
                    "(public web research; privacy-friendly; no private account access)"
                ),
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        results: List[SearchResult] = []
        seen: set[str] = set()

        try:
            with DDGS(timeout=20) as ddgs:
                for item in ddgs.text(query, max_results=max_results):
                    url = item.get("href") or item.get("url")

                    if not url:
                        continue

                    if url in seen:
                        continue

                    domain = domain_of(url)

                    if not self.include_social and domain in BLOCKED_DOMAINS:
                        continue

                    seen.add(url)

                    results.append(
                        SearchResult(
                            title=item.get("title", "Untitled"),
                            url=url,
                            snippet=item.get("body", ""),
                        )
                    )

        except Exception as exc:
            if self.verbose:
                print(f"[search failed] {exc}")
            return []

        if self.verbose:
            print(f"[search] Found {len(results)} result(s).")

        return results

    def fetch_text(self, result: SearchResult) -> SourceDoc:
        domain = domain_of(result.url)

        try:
            response = self.session.get(
                result.url, timeout=self.timeout, allow_redirects=True
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and not any(
                t in content_type
                for t in ["text/html", "text/plain", "application/xhtml"]
            ):
                return self._skipped_doc(
                    result, domain, f"Skipped non-text content type: {content_type}."
                )

            raw = response.content
            if looks_binary(raw):
                return self._skipped_doc(
                    result, domain, "Skipped binary-looking response."
                )

            response.encoding = response.encoding or "utf-8"
            html = response.text

        except Exception as exc:
            return SourceDoc(
                title=result.title,
                url=result.url,
                domain=domain,
                snippet=result.snippet,
                text=f"[Fetch failed: {exc}]",
                status="fetch_failed",
                niche_score=0.0,
                niche_reasons=["Could not fetch page."],
            )

        try:
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )

            if not extracted:
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "noscript", "svg", "form"]):
                    tag.decompose()
                extracted = soup.get_text(" ", strip=True)

            text = clean_text(extracted or "")

        except Exception as exc:
            return SourceDoc(
                title=result.title,
                url=result.url,
                domain=domain,
                snippet=result.snippet,
                text=f"[Parse failed: {exc}]",
                status="parse_failed",
                niche_score=0.0,
                niche_reasons=["Could not safely parse page text."],
            )

        doc = SourceDoc(
            title=result.title,
            url=result.url,
            domain=domain,
            snippet=result.snippet,
            text=text[:9000],
            status="ok" if len(text) >= 200 else "thin_text",
        )

        self.score_source(doc)
        doc.keywords = top_keywords(doc.text, limit=12)
        return doc

    def _skipped_doc(self, result: SearchResult, domain: str, reason: str) -> SourceDoc:
        return SourceDoc(
            title=result.title,
            url=result.url,
            domain=domain,
            snippet=result.snippet,
            text=f"[Skipped: {reason}]",
            status="skipped",
            niche_score=0.0,
            niche_reasons=[reason],
        )

    # Allusion favors niche, research-oriented sources.
    # The goal is discovery rather than popularity.
    # Scores are heuristic and intentionally biased toward
    # repositories, preprints, experimental work, and
    # early-stage technical projects.

    def score_source(self, doc: SourceDoc) -> None:
        text = f"{doc.title or ''} {doc.snippet or ''} {doc.text or ''}".lower()
        reasons: List[str] = []
        score = 0.0
        mainstream = 0.0

        term_hits = sorted({term for term in NICHE_TERMS if term in text})
        if term_hits:
            score += min(35, len(term_hits) * 4)
            reasons.append(
                f"Contains niche/technical language: {', '.join(term_hits[:7])}."
            )

        if doc.domain in HIGH_SIGNAL_DOMAINS or any(
            d in doc.domain for d in HIGH_SIGNAL_DOMAINS
        ):
            score += 28
            reasons.append("Comes from a research/open-source oriented source.")

        if doc.domain in MAINSTREAM_DOMAINS:
            mainstream += 35
            score -= 10
            reasons.append(
                "Mainstream source; useful for baseline, less useful as niche signal."
            )

        if re.search(r"\b(2024|2025|2026)\b", text):
            score += 10
            reasons.append("Mentions recent years, suggesting newer activity.")

        if re.search(
            r"\b(beta|alpha|prototype|demo|experiment|workshop|preprint)\b", text
        ):
            score += 12
            reasons.append("Signals early-stage or experimental work.")

        if re.search(
            r"\b(stars|forks|issues|pull requests|commits|repository)\b", text
        ):
            score += 8
            reasons.append(
                "Looks like an open-source project with inspectable activity."
            )

        if len(doc.text) < 800:
            score -= 8
            reasons.append("Short page text; signal may be weak.")

        if any(
            x in text for x in ["what is", "beginner", "introduction", "complete guide"]
        ):
            mainstream += 15

        doc.niche_score = max(0.0, min(100.0, score))
        doc.mainstream_score = max(0.0, min(100.0, mainstream))
        doc.niche_reasons = reasons or ["No strong niche signals detected."]

    def explore(self, topic: str, max_results: int = 10) -> AllusionReport:
        query = expand_query(topic)
        search_results = self.search(query, max_results=max_results)

        docs: List[SourceDoc] = []
        for idx, result in enumerate(search_results, start=1):
            if self.verbose:
                print(f"[fetch {idx}/{len(search_results)}] {result.url}")
            docs.append(self.fetch_text(result))
            time.sleep(self.delay_seconds)

        valid_docs = [d for d in docs if d.status in {"ok", "thin_text"}]

        themes = discover_themes(valid_docs)
        niche_signals = discover_niche_signals(valid_docs)
        mainstream_signals = discover_mainstream_signals(valid_docs)
        allusions = discover_allusions(valid_docs)
        # observations = generate_observations(valid_docs, topic)
        # analytical_findings = generate_analytical_findings(valid_docs, topic)
        observations = []
        analytical_findings = []
        cautions = build_cautions(docs)

        markdown = render_markdown_report(
            topic=topic,
            query=query,
            docs=docs,
            themes=themes,
            niche_signals=niche_signals,
            mainstream_signals=mainstream_signals,
            allusions=allusions,
            observations=observations,
            analytical_findings=analytical_findings,
            cautions=cautions,
        )

        return AllusionReport(
            topic=topic,
            generated_at=dt.datetime.now().isoformat(timespec="seconds"),
            query=query,
            sources=docs,
            themes=themes,
            niche_signals=niche_signals,
            mainstream_signals=mainstream_signals,
            allusions=allusions,
            observations=observations,
            analytical_findings=analytical_findings,
            cautions=cautions,
            markdown=markdown,
        )


def detect_query_mode(topic: str) -> str:
    text = topic.lower()

    mode_terms = {
        "ai_technical": [
            "ai",
            "agent",
            "agents",
            "llm",
            "machine learning",
            "deep learning",
            "reinforcement",
            "neuroevolution",
            "artificial life",
            "open ended",
            "open-ended",
            "autonomous",
            "algorithm",
            "model",
            "neural",
        ],
        "software": [
            "github",
            "open source",
            "open-source",
            "framework",
            "library",
            "toolkit",
            "api",
            "developer",
            "repository",
            "code",
            "package",
        ],
        "psychology_social": [
            "psychology",
            "psychological",
            "behavior",
            "behaviour",
            "identity",
            "motivation",
            "personality",
            "cognition",
            "mental health",
            "social",
            "development",
            "longitudinal",
        ],
        "biomedical_neuroscience": [
            "eeg",
            "brain",
            "neuroscience",
            "clinical",
            "medical",
            "biomarker",
            "patient",
            "diagnosis",
            "pac",
            "cfc",
            "neural signal",
            "electrophysiology",
        ],
        "business_policy": [
            "startup",
            "market",
            "industry",
            "policy",
            "regulation",
            "economics",
            "finance",
            "adoption",
            "risk",
            "governance",
        ],
    }

    scores = {
        mode: sum(1 for term in terms if term in text)
        for mode, terms in mode_terms.items()
    }

    best_mode, best_score = max(scores.items(), key=lambda item: item[1])

    return best_mode if best_score > 0 else "general"


def expand_query(topic: str) -> str:
    topic = topic.strip()
    mode = detect_query_mode(topic)

    expansions = {
        "ai_technical": (
            "github arxiv open source research benchmark framework "
            "experimental prototype implementation"
        ),
        "software": (
            "github documentation open source framework library "
            "implementation benchmark examples"
        ),
        "psychology_social": (
            "psychology research review longitudinal study meta analysis "
            "theory empirical evidence"
        ),
        "biomedical_neuroscience": (
            "biomedical research clinical study review meta analysis "
            "methods dataset evidence"
        ),
        "business_policy": (
            "industry report policy analysis market research adoption "
            "risk governance case study"
        ),
        "general": ("research review analysis study evidence overview"),
    }

    return f"{topic} {expansions.get(mode, expansions['general'])}"


def domain_of(url: str) -> str:
    return urlparse(url).netloc.lower().replace("www.", "")


def looks_binary(raw: bytes) -> bool:
    sample = raw[:2048]
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(32, 127)))
    non_text = sample.translate(None, text_chars)
    return len(non_text) / max(1, len(sample)) > 0.35


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([\w\)])([A-Z][a-z])", r"\1 \2", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) < 32]


def top_keywords(text: str, limit: int = 10) -> List[str]:
    counts = Counter(tokenize(text))
    return [word for word, _ in counts.most_common(limit)]


def discover_themes(docs: List[SourceDoc], limit: int = 8) -> List[str]:
    combined = " ".join(doc.text for doc in docs)
    keywords = top_keywords(combined, limit=limit)
    return [f"Repeated signal around `{kw}`." for kw in keywords] or [
        "No repeated themes detected."
    ]


def discover_niche_signals(docs: List[SourceDoc], limit: int = 7) -> List[str]:
    ranked = sorted(docs, key=lambda d: d.niche_score, reverse=True)
    signals = []
    for doc in ranked[:limit]:
        if doc.niche_score <= 0:
            continue
        reason = doc.niche_reasons[0] if doc.niche_reasons else "Possible niche source."
        signals.append(f"{doc.title} — niche score {doc.niche_score:.0f}/100. {reason}")
    return signals or ["No strong niche sources found. Try a more specific query."]


def discover_mainstream_signals(docs: List[SourceDoc], limit: int = 5) -> List[str]:
    ranked = sorted(docs, key=lambda d: d.mainstream_score, reverse=True)
    signals = []
    for doc in ranked[:limit]:
        if doc.mainstream_score <= 0:
            continue
        signals.append(
            f"{doc.title} — mainstream score {doc.mainstream_score:.0f}/100."
        )
    return signals or ["No obvious mainstream baseline sources detected."]


def discover_allusions(docs: List[SourceDoc], limit: int = 6):

    combined_keywords = []

    for doc in docs:
        combined_keywords.extend(
            [kw for kw in doc.keywords[:10] if kw not in IGNORED_ALLUSION_TERMS]
        )

    counts = Counter(combined_keywords)

    pairs = []

    for a, count_a in counts.most_common(20):
        for b, count_b in counts.most_common(20):

            if a >= b:
                continue

            if a in b or b in a:
                continue

            if a in IGNORED_ALLUSION_TERMS or b in IGNORED_ALLUSION_TERMS:
                continue

            pairs.append(
                (
                    count_a + count_b,
                    a,
                    b,
                )
            )

    pairs = sorted(pairs, reverse=True)[:limit]

    if not pairs:
        return ["No clear hidden connections detected yet. Try a narrower topic."]

    return [
        f"`{a}` repeatedly appears near `{b}`; "
        "this may be a useful connection to investigate."
        for _, a, b in pairs
    ]


def source_mentions(doc: SourceDoc, terms: Iterable[str]) -> bool:
    text = f"{doc.title or ''} {doc.snippet or ''} {doc.text or ''}".lower()
    return any(term.lower() in text for term in terms)


def evidence_titles(
    docs: List[SourceDoc], terms: Iterable[str], limit: int = 4
) -> List[str]:
    matches = [doc.title for doc in docs if source_mentions(doc, terms)]
    return matches[:limit]


# Observation Engine
#
# Converts source clusters into analyst-style observations.
# Observations must be evidence-backed and derived from
# fetched sources rather than generated speculation.


def generate_observations(
    docs: List[SourceDoc], topic: str, limit: int = 6
) -> List[Observation]:
    """
    Observation Engine v1.

    This turns search results into analyst-style observations by comparing
    source clusters instead of only reporting keywords.

    It is deliberately extractive:
    every observation must be backed by titles from the fetched sources.
    """
    if not docs:
        return [
            Observation(
                title="No usable source cluster found",
                insight="Allusion did not collect enough readable public sources to infer a pattern.",
                evidence=[],
                confidence=0.15,
                category="caution",
            )
        ]

    clusters = {
        "LLM-agent coordination": [
            "llm",
            "large language",
            "language model",
            "agentverse",
            "agentscope",
        ],
        "reinforcement learning": [
            "reinforcement",
            "rl",
            "policy",
            "reward",
            "multi-agent reinforcement",
        ],
        "simulation environments": [
            "simulation",
            "simulator",
            "environment",
            "benchmark",
            "scenario",
        ],
        "open-source tooling": [
            "github",
            "repository",
            "open-source",
            "install",
            "stars",
            "forks",
        ],
        "research surveys": ["survey", "review", "progress and challenges", "taxonomy"],
        "early-stage prototypes": [
            "prototype",
            "demo",
            "alpha",
            "beta",
            "experimental",
            "workshop",
            "preprint",
        ],
        "privacy/local-first": ["privacy", "local", "offline", "on-device", "private"],
        "neural/evolutionary methods": [
            "neuroevolution",
            "evolution",
            "genetic",
            "neural",
            "darwinian",
        ],
        "EEG/signal processing": [
            "eeg",
            "cross-frequency",
            "phase amplitude",
            "comodulogram",
            "pac",
            "cfc",
        ],
    }

    cluster_hits: dict[str, List[SourceDoc]] = {}
    for name, terms in clusters.items():
        hits = [doc for doc in docs if source_mentions(doc, terms)]
        if hits:
            cluster_hits[name] = hits

    observations: List[Observation] = []

    if cluster_hits:
        dominant_name, dominant_docs = max(
            cluster_hits.items(), key=lambda item: len(item[1])
        )
        confidence = min(0.95, 0.45 + len(dominant_docs) / max(1, len(docs)))
        observations.append(
            Observation(
                title=f"{dominant_name} appears to be the strongest signal",
                insight=(
                    f"Across the collected sources, `{dominant_name}` appears more often than the other detected clusters. "
                    "Treat this as the current center of gravity for the search topic."
                ),
                evidence=[d.title for d in dominant_docs[:4]],
                confidence=round(confidence, 2),
                category="dominant-pattern",
            )
        )

    llm_count = len(cluster_hits.get("LLM-agent coordination", []))
    rl_count = len(cluster_hits.get("reinforcement learning", []))
    sim_count = len(cluster_hits.get("simulation environments", []))
    open_count = len(cluster_hits.get("open-source tooling", []))

    if llm_count >= 3 and rl_count <= 1:
        observations.append(
            Observation(
                title="Possible gap: LLM agents dominate while RL appears sparse",
                insight=(
                    "The source set leans toward LLM-based agent frameworks, while reinforcement-learning-specific work appears less represented. "
                    "That gap may be worth investigating if the goal is to build adaptive agents or NPC-like systems."
                ),
                evidence=evidence_titles(
                    docs, ["llm", "large language", "agentverse", "agentscope"], limit=4
                ),
                confidence=0.72,
                category="opportunity",
            )
        )

    if sim_count >= 3 and open_count >= 3:
        observations.append(
            Observation(
                title="Open-source simulation infrastructure is a strong entry point",
                insight=(
                    "Multiple sources connect simulation environments with inspectable repositories or tooling. "
                    "This suggests the topic has enough public infrastructure to support prototypes, comparisons, and follow-up experiments."
                ),
                evidence=evidence_titles(
                    docs, ["github", "simulation", "simulator", "environment"], limit=4
                ),
                confidence=0.78,
                category="build-opportunity",
            )
        )

    if "research surveys" in cluster_hits:
        observations.append(
            Observation(
                title="Survey papers can anchor the research map",
                insight=(
                    "At least one survey or review-style source appeared. Use it as the backbone for building a cleaner taxonomy of the field."
                ),
                evidence=[d.title for d in cluster_hits["research surveys"][:4]],
                confidence=0.67,
                category="research-strategy",
            )
        )

    if "early-stage prototypes" in cluster_hits:
        observations.append(
            Observation(
                title="Early-stage work is present",
                insight=(
                    "The search surfaced prototype, demo, preprint, or experimental language. "
                    "That usually means the area is still moving and may contain niche opportunities."
                ),
                evidence=[d.title for d in cluster_hits["early-stage prototypes"][:4]],
                confidence=0.64,
                category="niche-signal",
            )
        )

    niche_docs = [d for d in docs if d.niche_score >= 70 and d.mainstream_score <= 15]
    if len(niche_docs) >= 2:
        observations.append(
            Observation(
                title="High-niche sources outweigh mainstream coverage",
                insight=(
                    "Several sources scored high for niche/research value while showing low mainstream baseline scores. "
                    "This is usually where Allusion is most useful: the topic is visible to builders and researchers before it is broadly popular."
                ),
                evidence=[
                    d.title
                    for d in sorted(
                        niche_docs, key=lambda x: x.niche_score, reverse=True
                    )[:4]
                ],
                confidence=0.8,
                category="niche-opportunity",
            )
        )

    if not observations:
        top = sorted(docs, key=lambda d: d.niche_score, reverse=True)[:3]
        observations.append(
            Observation(
                title="No strong cross-source pattern detected yet",
                insight=(
                    "The sources were readable, but Allusion did not find a confident repeated pattern. "
                    "Try a narrower query or add domain terms like `github`, `arxiv`, `framework`, or `benchmark`."
                ),
                evidence=[d.title for d in top],
                confidence=0.35,
                category="caution",
            )
        )

    priority = {
        "opportunity": 0,
        "build-opportunity": 1,
        "niche-opportunity": 2,
        "dominant-pattern": 3,
        "research-strategy": 4,
        "niche-signal": 5,
        "caution": 9,
    }
    observations = sorted(
        observations, key=lambda obs: (priority.get(obs.category, 8), -obs.confidence)
    )

    return observations[:limit]


# Domain routing system.
#
# Attempts to classify the topic into a known analytical
# domain so findings can be tailored to the source material.
#
# Current domains:
# - eeg
# - rl_evolution
# - multi_agent
# - general (fallback when no strong domain signals are detected)


def detect_topic_domain(topic: str, docs: List[SourceDoc]) -> str:
    text = " ".join(
        [topic] + [f"{d.title or ''} {d.snippet or ''} {d.text or ''}" for d in docs]
    ).lower()

    domain_terms = {
        "eeg": [
            "eeg",
            "brain",
            "neural",
            "electrophysiological",
            "pac",
            "cfc",
            "phase-amplitude",
            "cross-frequency",
        ],
        "rl_evolution": [
            "reinforcement",
            "reward",
            "policy",
            "evolution",
            "evolutionary",
            "open-ended",
            "neuroevolution",
            "self-evolving",
        ],
        "multi_agent": [
            "multi-agent",
            "agents",
            "autonomous agents",
            "llm agents",
            "agentic",
        ],
    }

    scores = {
        domain: sum(1 for term in terms if term in text)
        for domain, terms in domain_terms.items()
    }

    best_domain, best_score = max(scores.items(), key=lambda item: item[1])

    return best_domain if best_score > 0 else "general"


# Analytical Findings Engine
#
# Produces higher-level synthesis from retrieved sources.
# Unlike observations, findings attempt to identify
# implications, opportunities, and recurring research
# directions within a specific domain.


def generate_analytical_findings(
    docs: List[SourceDoc], topic: str
) -> List[AnalyticalFinding]:
    domain = detect_topic_domain(topic, docs)
    combined = " ".join(
        f"{d.title or ''} {d.snippet or ''} {d.text or ''}" for d in docs
    ).lower()

    def evidence_for(terms: List[str], limit: int = 4) -> List[str]:
        results = []
        for doc in docs:
            text = f"{doc.title or ''} {doc.snippet or ''} {doc.text or ''}".lower()
            if any(term in text for term in terms):
                results.append(doc.title)
        return results[:limit]

    findings: List[AnalyticalFinding] = []

    if domain == "eeg":
        findings.append(
            AnalyticalFinding(
                title="PAC/CFC is behaving like a feature-representation layer",
                finding="The sources repeatedly connect EEG, PAC/CFC, features, classification, and models.",
                implication="Your strongest pipeline framing is not just CFC visualization, but EEG decoding using CFC-derived heatmap features.",
                evidence=evidence_for(
                    [
                        "eeg",
                        "pac",
                        "cfc",
                        "phase-amplitude",
                        "classification",
                        "features",
                    ]
                ),
                confidence=0.84,
            )
        )

        findings.append(
            AnalyticalFinding(
                title="Statistical validation should be part of the pipeline",
                finding="PAC/CFC papers frequently discuss significance, inference, modulation index reliability, or statistical control.",
                implication="Add surrogate testing, significance masking, or confidence scoring before sending heatmaps into ML.",
                evidence=evidence_for(
                    [
                        "statistical",
                        "significance",
                        "inference",
                        "modulation index",
                        "surrogate",
                    ]
                ),
                confidence=0.78,
            )
        )

    elif domain == "rl_evolution":
        findings.append(
            AnalyticalFinding(
                title="The field is moving from trained agents toward self-evolving agents",
                finding="The source set repeatedly connects open-ended learning, evolution, self-improvement, autonomous agents, and research systems.",
                implication="The highest-upside direction is not just RL training, but agents that modify strategies, tools, or behaviors over time.",
                evidence=evidence_for(
                    [
                        "open-ended",
                        "evolution",
                        "self-evolving",
                        "self-improving",
                        "autonomous",
                    ]
                ),
                confidence=0.86,
            )
        )

        findings.append(
            AnalyticalFinding(
                title="Classical RL is not the center of gravity",
                finding="The results emphasize open-ended discovery, evolutionary systems, and autonomous agent research more than PPO/Q-learning-style methods.",
                implication="Search and build around open-ended evolution, agent self-improvement, and artificial life rather than only reward optimization.",
                evidence=evidence_for(
                    [
                        "evolution",
                        "open-ended",
                        "autonomous",
                        "multi-agent",
                        "discovery",
                    ]
                ),
                confidence=0.79,
            )
        )

    elif domain == "multi_agent":
        findings.append(
            AnalyticalFinding(
                title="Multi-agent work is clustering around coordination frameworks",
                finding="The source set repeatedly connects agents, frameworks, simulation, tools, and autonomous workflows.",
                implication="A strong project angle is comparing how different frameworks support memory, planning, coordination, and environment feedback.",
                evidence=evidence_for(
                    [
                        "multi-agent",
                        "framework",
                        "coordination",
                        "simulation",
                        "autonomous",
                    ]
                ),
                confidence=0.78,
            )
        )

    else:
        findings.append(
            AnalyticalFinding(
                title="No confident domain-specific synthesis detected",
                finding="Allusion found sources, but the topic did not match a known analytical domain strongly enough.",
                implication="Use a more specific query with method + domain + task.",
                evidence=[doc.title for doc in docs[:3]],
                confidence=0.35,
            )
        )

    return findings


def build_cautions(docs: List[SourceDoc]) -> List[str]:
    cautions = []
    failed = [d for d in docs if d.status == "fetch_failed"]
    parsed = [d for d in docs if d.status == "parse_failed"]
    skipped = [d for d in docs if d.status == "skipped"]

    if failed:
        cautions.append(f"{len(failed)} source(s) could not be fetched.")
    if parsed:
        cautions.append(f"{len(parsed)} source(s) could not be parsed safely.")
    if skipped:
        cautions.append(
            f"{len(skipped)} non-text/binary/social source(s) were skipped."
        )
    if len([d for d in docs if d.status in {"ok", "thin_text"}]) < 4:
        cautions.append(
            "Small usable source set; conclusions should be treated as exploratory."
        )

    cautions.append(
        "This agent only analyzes public pages and should not be treated as authoritative research."
    )
    return cautions


# Report Renderer
#
# Converts structured Allusion output into a human-readable
# research brief suitable for archiving or sharing.


def render_markdown_report(
    topic: str,
    query: str,
    docs: List[SourceDoc],
    themes: List[str],
    niche_signals: List[str],
    mainstream_signals: List[str],
    allusions: List[str],
    observations: List[Observation],
    analytical_findings: List[AnalyticalFinding],
    cautions: List[str],
) -> str:
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Allusion Report: {topic}",
        "",
        f"Generated: {generated}",
        "",
        "## Search Frame",
        "",
        f"Query used: `{query}`",
        "",
        "**Research Note:**",
        "",
        "Allusion generates exploratory observations from public sources.",
        "Verify findings against the linked sources before relying on them.",
        "",
    ]

    # lines += [
    #     "## Analytical Findings",
    #     "",
    # ]

    # for finding in analytical_findings:
    #     lines += [
    #         f"### {finding.title}",
    #         f"- Confidence: {finding.confidence:.2f}",
    #         f"- Finding: {finding.finding}",
    #         f"- Implication: {finding.implication}",
    #     ]

    #     if finding.evidence:
    #         lines.append("- Evidence:")
    #         for item in finding.evidence:
    #             lines.append(f"  - {item}")

    #     lines.append("")

    lines += [
        "## Observations",
        "",
    ]

    for obs in observations:
        lines += [
            f"### {obs.title}",
            f"- Category: {obs.category}",
            f"- Confidence: {obs.confidence:.2f}",
            f"- Insight: {obs.insight}",
        ]

        if obs.evidence:
            lines.append("- Evidence:")
            for item in obs.evidence:
                lines.append(f"  - {item}")

        lines.append("")

    lines += [
        "## What Looks Niche",
        "",
    ]

    for signal in niche_signals:
        lines.append(f"- {signal}")

    lines += [
        "",
        "## Mainstream Baseline",
        "",
    ]

    for signal in mainstream_signals:
        lines.append(f"- {signal}")

    lines += [
        "",
        "## Allusions in the Machine",
        "",
    ]

    for item in allusions:
        lines.append(f"- {item}")

    lines += [
        "",
        "## Repeated Themes",
        "",
    ]

    for theme in themes:
        lines.append(f"- {theme}")

    lines += [
        "",
        "## Ranked Sources",
        "",
    ]

    ranked_docs = sorted(docs, key=lambda d: d.niche_score, reverse=True)

    for i, doc in enumerate(ranked_docs, start=1):
        lines += [
            f"### {i}. {doc.title}",
            f"- URL: {doc.url}",
            f"- Domain: {doc.domain}",
            f"- Status: {doc.status}",
            f"- Niche score: {doc.niche_score:.0f}/100",
            f"- Mainstream score: {doc.mainstream_score:.0f}/100",
            f"- Keywords: {', '.join(doc.keywords[:8]) if doc.keywords else 'none'}",
            f"- Why it matters: {doc.niche_reasons[0] if doc.niche_reasons else 'No reason generated.'}",
            "",
        ]

    lines += [
        "## Cautions",
        "",
    ]

    for caution in cautions:
        lines.append(f"- {caution}")

    lines += [
        "",
        "## Privacy Note",
        "",
        "Allusion used only the topic provided by the user and public webpages. "
        "It did not access private accounts, emails, files, contacts, calendars, or content behind logins.",
        "",
    ]

    return "\n".join(lines)


def save_report(report: AllusionReport, topic: str) -> Path:
    archive = Path("research_archive")
    archive.mkdir(exist_ok=True)

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", topic.lower()).strip("-")[:60] or "report"
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    run_dir = archive / f"{timestamp}-{slug}"
    run_dir.mkdir(exist_ok=True)

    markdown_path = run_dir / "report.md"
    json_path = run_dir / "data.json"

    markdown_path.write_text(report.markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Allusion v2: privacy-friendly public-web exploration agent."
    )
    parser.add_argument("topic", help="Topic to investigate.")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.75)
    parser.add_argument(
        "--include-social",
        action="store_true",
        help="Allow social/video domains in search results.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print search/fetch progress."
    )
    args = parser.parse_args()

    agent = AllusionAgent(
        delay_seconds=args.delay,
        include_social=args.include_social,
        verbose=args.verbose,
    )
    report = agent.explore(args.topic, max_results=args.max_results)
    path = save_report(report, args.topic)

    print(report.markdown)
    print(f"\nSaved report to: {path}")


if __name__ == "__main__":
    main()
