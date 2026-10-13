"""Tiny keyword retriever: split a document into paragraphs and rank them
by overlap with the question's keywords. No embeddings, no network —
just the standard library, so it runs anywhere."""

import re

STOPWORDS = frozenset(
    """
    a an the and or but of to in on for with by from at as is are was were be
    been being it its this that these those what which who whom whose when
    where why how does do did can could should would will shall may might must
    have has had having not no yes if then than so such only also very per
    """.split()
)


def split_paragraphs(text):
    """Split text into non-empty paragraphs (blank-line separated)."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_sentences(text):
    """Split text into sentences on '. '/('! ')/('? ') boundaries.

    Naive but dependency-free; good enough for prose documents.
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“'])", text)
    return [s.strip() for s in parts if s.strip()]


def keywords(question):
    """Extract content keywords from a question: lowercase words longer
    than 3 chars that are not stopwords."""
    words = re.findall(r"[a-z0-9]+", question.lower())
    return [w for w in words if len(w) > 3 and w not in STOPWORDS]


def stem(word):
    """Naive suffix stripper: removes a single trailing 'ing', 'ed' or 's'
    from longer words so 'terminating' matches 'terminate'/'termination'
    and 'invoices' matches 'invoice'. Deliberately dumb and documented."""
    if len(word) > 6 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 5 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s"):
        return word[:-1]
    return word


def score_paragraph(paragraph, kws):
    """Return (hit_count, min_span).

    hit_count: how many question keywords occur in the paragraph
    (substring match on stemmed keywords).
    min_span: the smallest character span covering one occurrence of every
    matched keyword — a tighter span means the paragraph is *about* the
    question, not just mentioning the words far apart. Used to break ties.
    """
    low = paragraph.lower()
    positions = []
    for kw in kws:
        skw = stem(kw)
        pos = low.find(skw)
        if pos != -1:
            # Prefer the full keyword when both match; stem is a fallback.
            full = low.find(kw)
            positions.append(full if full != -1 else pos)
    if not positions:
        return (0, float("inf"))
    return (len(positions), max(positions) - min(positions))


def rank(chunks, question, top_k=2):
    """Return [(1-based index, chunk)] of the top_k chunks most relevant
    to the question. Chunks are usually sentences: small, focused context
    keeps tiny local models accurate.

    Primary key: keyword hit count. Ties break toward the chunk whose
    matched keywords sit closest together (tighter span = more focused),
    then document order. Chunks with zero hits are dropped when anything
    matched; otherwise we fall back to document order.
    """
    kws = keywords(question)
    if not kws:
        return [(i + 1, c) for i, c in enumerate(chunks[:top_k])]
    scored = []
    for i, c in enumerate(chunks):
        count, span = score_paragraph(c, kws)
        scored.append((count, span, i, c))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    matched = [t for t in scored if t[0] > 0]
    pool = matched if matched else scored
    return [(i + 1, c) for _, _, i, c in pool[:top_k]]


def build_excerpts(ranked):
    """Format ranked chunks as plain text for the prompt.

    Deliberately no '[n]' numbering: tiny local models get confused by
    index prefixes. The CLI tells the user which sentences were used.
    """
    return "\n\n".join(para for _, para in ranked)
