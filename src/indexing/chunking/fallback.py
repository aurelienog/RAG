from ...domain import Chunk


def split_lines(
    text: str,
    file_path: str,
    start_offset: int,
    max_chunk_size: int,
    kind: str,
) -> list[Chunk]:
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
