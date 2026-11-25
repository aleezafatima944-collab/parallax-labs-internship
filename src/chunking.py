"""Recursive character chunking for the clean corpus."""

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text, max_len, separators):
    """Break text into pieces no longer than max_len, preferring natural breaks."""
    if len(text) <= max_len:
        return [text]
    if not separators:
        return [text[i : i + max_len] for i in range(0, len(text), max_len)]

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    pieces = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            part += sep
        if len(part) > max_len:
            pieces.extend(_split(part, max_len, rest))
        else:
            pieces.append(part)
    return pieces


def _overlap_tail(text, overlap):
    """Return the last `overlap` characters, starting at a word boundary."""
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split one document into overlapping chunks of at most chunk_size characters."""
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size")
    if not text or not text.strip():
        return []

    pieces = _split(text, chunk_size - overlap, SEPARATORS)

    chunks = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            current = _overlap_tail(current, overlap) + piece
        else:
            current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_documents(df, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Turn a DataFrame of documents into a list of chunk records."""
    records = []
    for row in df.itertuples(index=False):
        for i, chunk in enumerate(chunk_text(row.text, chunk_size, overlap)):
            records.append(
                {
                    "chunk_id": f"{row.id}_{i}",
                    "doc_id": row.id,
                    "title": row.title,
                    "chunk_index": i,
                    "text": chunk,
                }
            )
    return records