"""
Build a KB source file from the PubMed Central Open Access Subset.

This script uses official NCBI/PMC retrieval endpoints to fetch article
metadata and text, then writes a JSON file with entries shaped for the
knowledge-base builder:

[
  {"text": "...", "source": "...", "topic": "..."},
  ...
]

Examples:
    uv run python scripts/build_pmc_source_file.py \
      --query "radiology" \
      --max_articles 200 \
      --output_file data/knowledge/pmc_radiology.json

    uv run python scripts/build_pmc_source_file.py \
      --query "chest x-ray pneumonia" \
      --topic radiology \
      --max_articles 100 \
      --output_file data/knowledge/pmc_chest_xray.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from vqa_med.config import config


PMC_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OA_SEARCH = f"{PMC_EUTILS_BASE}/esearch.fcgi"
PMC_EFETCH = f"{PMC_EUTILS_BASE}/efetch.fcgi"


TOPIC_KEYWORDS = {
    "radiology": {
        "x-ray", "xray", "radiograph", "radiographic", "ct", "computed tomography",
        "mri", "magnetic resonance", "ultrasound", "sonography", "pet", "imaging",
        "image", "scan", "opacity", "lesion", "nodule", "consolidation", "contrast",
        "flair", "diffusion", "dwi", "axial", "coronal", "sagittal",
    },
    "anatomy": {
        "anatomy", "anatomical", "organ", "tissue", "artery", "vein", "nerve", "muscle",
        "lung", "heart", "liver", "kidney", "brain", "spine", "abdomen", "thorax",
    },
    "pathology": {
        "pathology", "disease", "syndrome", "lesion", "tumor", "cancer", "infection",
        "inflammation", "edema", "hemorrhage", "fracture", "pneumonia", "embolism",
    },
    "clinical_basics": {
        "clinical", "diagnosis", "symptom", "treatment", "therapy", "patient", "history",
        "risk", "prognosis", "management", "guideline",
    },
}

LOW_VALUE_TERMS = {
    "cost-effectiveness", "quality-adjusted", "questionnaire", "survey", "randomized trial",
    "protocol", "recruitment", "ethics approval", "informed consent", "registry",
}


@dataclass
class ArticleRecord:
    text: str
    source: str
    topic: str


def fetch_url(url: str, timeout: int = 60) -> str:
    request = Request(url, headers={"User-Agent": "vqa-med/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def text_from_element(element: Optional[ET.Element]) -> str:
    if element is None:
        return ""
    return " ".join(part.strip() for part in element.itertext() if part and part.strip())


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def topic_relevance_score(text: str, topic: str) -> int:
    keywords = TOPIC_KEYWORDS.get(topic.lower())
    if not keywords:
        return 1

    lower = text.lower()
    return sum(1 for keyword in keywords if keyword in lower)


def has_low_value_terms(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in LOW_VALUE_TERMS)


def search_pmc_ids(query: str, max_articles: int, api_key: Optional[str] = None) -> List[str]:
    params = [
        "db=pmc",
        f"term={quote_plus(query)}",
        f"retmax={max_articles}",
        "retmode=json",
        "sort=relevance",
    ]
    if api_key:
        params.append(f"api_key={quote_plus(api_key)}")

    url = f"{PMC_OA_SEARCH}?{'&'.join(params)}"
    payload = fetch_url(url)
    data = json.loads(payload)
    return data.get("esearchresult", {}).get("idlist", [])


def split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitting without external dependencies.
    chunks = []
    current = []
    for token in text.split():
        current.append(token)
        if token.endswith((".", "?", "!")) and len(" ".join(current).split()) >= 8:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def extract_article_records(xml_text: str, topic: str) -> List[ArticleRecord]:
    root = ET.fromstring(xml_text)
    namespace = {"pmc": root.tag.split("}")[0].strip("{") if "}" in root.tag else ""}

    records: List[ArticleRecord] = []
    article_title = normalize_whitespace(text_from_element(root.find(".//article-title", namespace)))
    journal_title = normalize_whitespace(text_from_element(root.find(".//journal-title", namespace)))
    pmcid = normalize_whitespace(text_from_element(root.find(".//article-id[@pub-id-type='pmc']", namespace)))
    pmid = normalize_whitespace(text_from_element(root.find(".//article-id[@pub-id-type='pmid']", namespace)))

    source_parts = [part for part in [pmcid or None, pmid or None, article_title or None, journal_title or None] if part]
    source = " | ".join(source_parts) if source_parts else "PMC Open Access Subset"

    abstract = normalize_whitespace(text_from_element(root.find(".//abstract", namespace)))
    if abstract:
        for sentence in split_sentences(abstract):
            if len(sentence.split()) >= 8:
                records.append(ArticleRecord(text=sentence, source=source, topic=topic))

    body_sections = root.findall(".//body//sec", namespace)
    if not body_sections:
        # Fall back to paragraphs in the full text body.
        for paragraph in root.findall(".//body//p", namespace):
            paragraph_text = normalize_whitespace(text_from_element(paragraph))
            if len(paragraph_text.split()) >= 8:
                records.append(ArticleRecord(text=paragraph_text, source=source, topic=topic))
        return records

    for section in body_sections:
        section_title = normalize_whitespace(text_from_element(section.find("title", namespace)))
        for paragraph in section.findall(".//p", namespace):
            paragraph_text = normalize_whitespace(text_from_element(paragraph))
            if len(paragraph_text.split()) < 8:
                continue
            if section_title:
                paragraph_text = f"{section_title}: {paragraph_text}"
            records.append(ArticleRecord(text=paragraph_text, source=source, topic=topic))

    return records


def fetch_pmc_article(pmc_id: str, api_key: Optional[str] = None) -> Optional[str]:
    params = [
        "db=pmc",
        f"id={quote_plus(pmc_id)}",
        "retmode=xml",
    ]
    if api_key:
        params.append(f"api_key={quote_plus(api_key)}")

    url = f"{PMC_EFETCH}?{'&'.join(params)}"
    try:
        return fetch_url(url)
    except (HTTPError, URLError, TimeoutError):
        return None


def build_records(
    query: str,
    max_articles: int,
    topic: str,
    api_key: Optional[str] = None,
    delay_seconds: float = 0.34,
    strict_topic_filter: bool = False,
    min_topic_hits: int = 1,
    exclude_low_value: bool = True,
) -> List[dict]:
    pmc_ids = search_pmc_ids(query=query, max_articles=max_articles, api_key=api_key)
    output: List[dict] = []
    seen_text = set()

    for pmc_id in pmc_ids:
        xml_text = fetch_pmc_article(pmc_id, api_key=api_key)
        if not xml_text:
            continue

        try:
            records = extract_article_records(xml_text, topic=topic)
        except ET.ParseError:
            continue

        for record in records:
            normalized_text = normalize_whitespace(record.text)

            if exclude_low_value and has_low_value_terms(normalized_text):
                continue

            if strict_topic_filter:
                score = topic_relevance_score(normalized_text, topic)
                if score < min_topic_hits:
                    continue

            if normalized_text.lower() in seen_text:
                continue
            seen_text.add(normalized_text.lower())
            output.append({
                "text": normalized_text,
                "source": record.source,
                "topic": record.topic,
            })

        time.sleep(delay_seconds)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a KB source file from PMC Open Access articles")
    parser.add_argument("--query", type=str, default="radiology", help="PMC search query")
    parser.add_argument("--topic", type=str, default="radiology", help="Topic label to write into metadata")
    parser.add_argument("--max_articles", type=int, default=100, help="Maximum number of PMC articles to fetch")
    parser.add_argument("--output_file", type=str, default=None, help="JSON output file path")
    parser.add_argument("--api_key", type=str, default=None, help="Optional NCBI API key")
    parser.add_argument(
        "--strict_topic_filter",
        action="store_true",
        help="Keep only passages strongly matching the selected topic keywords",
    )
    parser.add_argument(
        "--min_topic_hits",
        type=int,
        default=2,
        help="Minimum keyword hits required when strict topic filtering is enabled",
    )
    parser.add_argument(
        "--allow_low_value_terms",
        action="store_true",
        help="Allow passages with low-value research-process terms",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = Path(args.output_file) if args.output_file else config.paths.data_root / "knowledge" / "pmc_source.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building KB Source File from PMC Open Access")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Topic: {args.topic}")
    print(f"Max articles: {args.max_articles}")
    print(f"Output: {output_file}")
    print(f"Strict topic filter: {args.strict_topic_filter}")
    if args.strict_topic_filter:
        print(f"Min topic keyword hits: {args.min_topic_hits}")

    records = build_records(
        query=args.query,
        max_articles=args.max_articles,
        topic=args.topic,
        api_key=args.api_key,
        strict_topic_filter=args.strict_topic_filter,
        min_topic_hits=max(1, args.min_topic_hits),
        exclude_low_value=not args.allow_low_value_terms,
    )

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    print(f"✓ Wrote {len(records)} records to {output_file}")


if __name__ == "__main__":
    main()