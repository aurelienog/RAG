import re

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "in",
    "on", "to", "for", "with", "by", "as", "is", "it", "this", "that",
    "these", "those", "be", "are", "was", "were", "from", "at", "into",
    "your", "you", "we", "our", "i", "me", "my", "do", "does", "did",
    "have", "has", "had", "can", "will", "would", "should",
}

# _CAMEL_REGEX = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_CAMEL_REGEX = re.compile(r"(?<=[a-z0-9])(?=[A-Z])" r"|" r"(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT_REGEX = re.compile(r"[^A-Za-z0-9]+")
_TOKEN_REGEX = re.compile(r"[A-Za-z_][A-Za-z0-9_]*" r"|" r"\d+(?:\.\d+)?")


class Tokenizer:
    """Tokenize code and prose for lexical retrieval."""
    MIN_TOKEN_LENGTH = 2

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Tokenize text for lexical retrieval.

        The tokenizer:

        - lowercases tokens;
        - preserves complete identifiers;
        - decomposes camelCase/PascalCase identifiers;
        - decomposes snake_case and kebab-case identifiers;
        - preserves useful numeric tokens;
        - removes very short tokens;
        - removes stopwords from ordinary prose;
        - keeps code-like identifiers intact;
        - preserves token order.

        Args:
            text: Text to tokenize.

        Returns:
            Normalized retrieval tokens.
        """
        if not text:
            return []

        tokens: list[str] = []

        for match in _TOKEN_REGEX.finditer(text):
            raw = match.group(0)

            if not raw:
                continue

            # Underscore is meaningful in identifiers, but it is not useful
            # as an independent lexical token.
            normalized = raw.lower()

            # Detect whether this looks like an identifier.
            is_identifier = (
                "_" in raw
                or any(char.isupper() for char in raw[1:])
            )

            # Preserve the complete identifier.
            if is_identifier:
                if cls._valid_token(normalized):
                    tokens.append(normalized)

                # Add decomposed identifier parts.
                parts = cls._split_identifier(raw)

                for part in parts:
                    normalized_part = part.lower()

                    if cls._valid_token(normalized_part):
                        tokens.append(normalized_part)

                continue

            # Normal prose / ordinary token.
            if not cls._valid_token(normalized):
                continue

            # Stopwords are removed from normal prose.
            if normalized in _STOPWORDS:
                continue

            tokens.append(normalized)

        return tokens

    @classmethod
    def tokenize_batch(
        cls,
        texts: list[str],
    ) -> list[list[str]]:
        """Tokenize multiple texts.

        Args:
            texts: Input texts.

        Returns:
            One token list per input text.
        """
        return [cls.tokenize(text) for text in texts]

    @classmethod
    def _split_identifier(
        cls,
        identifier: str,
    ) -> list[str]:
        """Split an identifier into lexical components.

        Examples:

            getUserById
                -> ["get", "User", "By", "Id"]

            HTTPServer
                -> ["HTTP", "Server"]

            get_user_by_id
                -> ["get", "user", "by", "id"]

            parse-json-response
                -> ["parse", "json", "response"]
        """
        # First split snake_case / kebab-case / similar identifiers.
        parts = re.split(r"[_\-\s]+", identifier)

        result: list[str] = []

        for part in parts:
            if not part:
                continue

            # Then split CamelCase/PascalCase.
            camel_parts = _CAMEL_REGEX.split(part)

            for camel_part in camel_parts:
                if camel_part:
                    result.append(camel_part)

        return result

    @classmethod
    def _valid_token(
        cls,
        token: str,
    ) -> bool:
        """Return whether a token is useful for lexical retrieval."""
        if len(token) < cls.MIN_TOKEN_LENGTH:
            return False

        # Ignore tokens consisting exclusively of underscores.
        if token.strip("_") == "":
            return False

        return True
