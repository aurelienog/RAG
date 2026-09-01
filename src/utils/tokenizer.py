import re

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "in",
    "on", "to", "for", "with", "by", "as", "is", "it", "this", "that",
    "these", "those", "be", "are", "was", "were", "from", "at", "into",
    "your", "you", "we", "our", "i", "me", "my", "do", "does", "did",
    "have", "has", "had", "can", "will", "would", "should",
}

_CAMEL_REGEX = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT_REGEX = re.compile(r"[^A-Za-z0-9]+")


class Tokenizer:
    """Simple tokenizer tuned for code and prose retrieval."""

    @staticmethod
    def tokenize(text: str) -> list[str]:

        if not text:
            return []

        tokens: list[str] = []
        for part in _SPLIT_REGEX.split(text):
            if not part:
                continue
            for subpart in _CAMEL_REGEX.split(part):
                token = subpart.lower()
                if len(token) < 2:
                    continue
                if token in _STOPWORDS:
                    continue
                tokens.append(token)
        return tokens

    @staticmethod
    def tokenize_batch(texts: list[str]) -> list[list[str]]:
        return [Tokenizer.tokenize(text) for text in texts]
