from ...domain import Chunk


def split_lines(
    text: str,
    file_path: str,
    start_offset: int,
    max_chunk_size: int,
    kind: str,
) -> list[Chunk]:
    """Split text into chunks while preserving complete lines when possible.

    Lines that exceed ``max_chunk_size`` are split into smaller chunks using
    ``hard_split``. Each returned chunk contains its absolute character
    offsets within the original source file.

    Args:
        text: Source text to split into chunks.
        file_path: Path of the source file containing the text.
        start_offset: Absolute character offset where ``text`` begins in the
            source file.
        max_chunk_size: Maximum number of characters allowed in each chunk.
        kind: Type or category assigned to each generated chunk.

    Returns:
        A list of chunks created from the input text.
    """
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_length = 0

    absolute_cursor = start_offset
    chunk_start_absolute = start_offset

    for line in text.splitlines(keepends=True):
        line_length = len(line)

        if line_length > max_chunk_size:

            if current_lines:
                chunks.append(
                    create_chunk(current_lines, file_path, chunk_start_absolute, kind)
                )
                current_lines = []
                current_length = 0

            chunks.extend(
                hard_split(line, file_path, absolute_cursor, max_chunk_size, kind)
            )

            absolute_cursor += line_length
            chunk_start_absolute = absolute_cursor
            continue

        if current_length + line_length > max_chunk_size:

            chunks.append(
                create_chunk(current_lines, file_path, chunk_start_absolute, kind)
            )
            current_lines = [line]
            current_length = line_length
            chunk_start_absolute = absolute_cursor
        else:
            current_lines.append(line)
            current_length += line_length

        absolute_cursor += line_length

    if current_lines:
        chunks.append(
            create_chunk(current_lines, file_path, chunk_start_absolute, kind)
        )

    return chunks


def hard_split(
    text: str,
    file_path: str,
    start_offset: int,
    max_chunk_size: int,
    kind: str,
) -> list[Chunk]:
    """Split text into fixed-size chunks without preserving line boundaries.

    The input is divided into consecutive slices of at most
    ``max_chunk_size`` characters. Each chunk receives absolute character
    offsets relative to the original source file.

    Args:
        text: Text to split into fixed-size chunks.
        file_path: Path of the source file containing the text.
        start_offset: Absolute character offset where ``text`` begins in the
            source file.
        max_chunk_size: Maximum number of characters allowed in each chunk.
        kind: Type or category assigned to each generated chunk.

    Returns:
        A list of fixed-size chunks covering the entire input text.
    """

    chunks: list[Chunk] = []

    for offset in range(0, len(text), max_chunk_size):
        chunk_text = text[offset:offset + max_chunk_size]
        start = start_offset + offset
        end = start + len(chunk_text)

        chunks.append(
            Chunk(
                id=f"{file_path}_{start}_{end}",
                file_path=file_path,
                text=chunk_text,
                start=start,
                end=end,
                kind=kind,
            )
        )

    return chunks


def create_chunk(
    lines: list[str],
    file_path: str,
    start: int,
    kind: str,
) -> Chunk:
    """Create a chunk from a sequence of source lines.

    The chunk end offset is calculated from its starting offset and the
    length of the combined text.

    Args:
        lines: Source lines to combine into a single chunk.
        file_path: Path of the source file containing the lines.
        start: Absolute character offset where the chunk begins.
        kind: Type or category assigned to the chunk.

    Returns:
        A ``Chunk`` containing the combined lines and their source offsets.
    """

    text = "".join(lines)
    end = start + len(text)

    return Chunk(
        id=f"{file_path}_{start}_{end}",
        file_path=file_path,
        text=text,
        start=start,
        end=end,
        kind=kind,
    )
