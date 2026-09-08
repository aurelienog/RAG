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
    """Tokenize code and prose for lexical retrieval."""

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Convert text into normalized retrieval tokens.

        The tokenizer separates punctuation and CamelCase identifiers,
        converts tokens to lowercase, removes short tokens and English
        stopwords, and preserves the remaining tokens in their original
        order.

        Args:
            text: Text to tokenize.

        Returns:
            A list of normalized tokens. An empty list is returned when the
            input text is empty or contains no valid tokens.
        """

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
        """Tokenize multiple texts using the same tokenizer.

        Args:
            texts: List of texts to tokenize.

        Returns:
            A list of token lists, with one token list corresponding to each
            input text.
        """
        return [Tokenizer.tokenize(text) for text in texts]
