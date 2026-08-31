from pydantic import BaseModel

from ..models import MinimalSource


class Chunk(BaseModel):
    """
    A source fragment produced during indexing.

    Character offsets use the half-open interval [start, end),
    so `text == source[start:end]`.
    """

    id: str
    file_path: str
    text: str
    start: int
    end: int
    kind: str

    def to_minimal_source(self) -> MinimalSource:
        return MinimalSource(
            file_path=self.file_path,
            first_character_index=self.start,
            last_character_index=self.end,
        )
