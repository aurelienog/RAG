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
    """Tokenize code and prose for lexical retrieval.

    Provides high-fidelity token extraction designed specifically for mixing programming
    languages and standard prose. It preserves compound identifiers while simultaneously
    decomposing them into individual sub-tokens to maximize lexical search recall.

    Attributes:
        MIN_TOKEN_LENGTH (int): Minimum character size threshold required for any token
            to be considered valid. Defaults to 2.
    """
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
        """Tokenize multiple text strings concurrently or sequentially.

        Processes an array of source strings into structural matrices of normalized tokens.

        Args:
            texts (list[str]): A list containing multiple separate input strings.

        Returns:
            list[list[str]]: A list of token arrays, maintaining a 1:1 mapping with the
                input text indices.
        """
        return [cls.tokenize(text) for text in texts]

    @classmethod
    def _split_identifier(
        cls,
        identifier: str,
    ) -> list[str]:
        """Split a code identifier into its fundamental lexical components.

        Decomposes complex structural names written across multiple styling paradigms
        such as snake_case, kebab-case, or CamelCase.

        Args:
            identifier (str): The raw compound identifier string extracted from the text.

        Returns:
            list[str]: The sub-component word fragments extracted from the identifier.

        Examples:
            >>> Tokenizer._split_identifier("getUserById")
            ['get', 'User', 'By', 'Id']

            >>> Tokenizer._split_identifier("HTTPServer")
            ['HTTP', 'Server']

            >>> Tokenizer._split_identifier("get_user_by_id")
            ['get', 'user', 'by', 'id']

            >>> Tokenizer._split_identifier("parse-json-response")
            ['parse', 'json', 'response']
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
        """Return whether a token is useful for lexical retrieval.

        Checks the token against structural constraints, such as minimum length
        boundaries and filtering out noise like lone underscore characters.

        Args:
            token (str): The normalized individual token string to evaluate.

        Returns:
            bool: True if the token meets validation criteria for indexing, False otherwise.
        """
        if len(token) < cls.MIN_TOKEN_LENGTH:
            return False

        # Ignore tokens consisting exclusively of underscores.
        if token.strip("_") == "":
            return False

        return True
