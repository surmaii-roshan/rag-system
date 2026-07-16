"""
generate/citation.py — Citation extraction and phantom citation detection.

Phantom citations: the LLM cites "[Source: doc.pdf, Page: 5]" but
doc.pdf page 5 was never provided in the context. This module catches those.
"""

import re
from typing import List, Tuple

from retrieve.vector_search import SearchResult
from utils.logger import get_logger

log = get_logger(__name__)

# Matches: [Source: filename.pdf, Page: 3] or [Source: filename, Page: 12]
_CITATION_RE = re.compile(
    r"\[Source:\s*([^\],]+?),\s*Page:\s*(\d+)\]",
    re.IGNORECASE,
)


def extract_citations(answer: str) -> List[Tuple[str, int]]:
    """
    Extract all (filename, page_number) citations from an answer string.

    Args:
        answer: Raw LLM-generated answer text.

    Returns:
        List of (filename, page_number) tuples found in the answer.
    """
    matches = _CITATION_RE.findall(answer)
    return [(filename.strip(), int(page)) for filename, page in matches]


def verify_citations(
    answer: str,
    provided_chunks: List[SearchResult],
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Verify citations in an answer against the actually-provided chunks.

    Args:
        answer: Raw LLM answer text containing [Source: ...] references.
        provided_chunks: The chunks that were included in the prompt.

    Returns:
        Tuple of:
          - valid_citations:   citations that match a provided chunk
          - phantom_citations: citations with no matching provided chunk
    """
    cited = extract_citations(answer)

    provided_sources: set[Tuple[str, int]] = {
        (c.source_file, c.page_number) for c in provided_chunks
    }

    valid = []
    phantom = []

    for filename, page in cited:
        if (filename, page) in provided_sources:
            valid.append((filename, page))
        else:
            phantom.append((filename, page))
            log.warning(f"Phantom citation detected: [{filename}, Page {page}]")

    if phantom:
        log.warning(f"{len(phantom)} phantom citation(s) in answer")

    return valid, phantom