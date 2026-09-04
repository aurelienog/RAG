def build_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build the prompt used to generate a grounded answer.

    Args:
        question: User question.
        context: Retrieved source text.

    Returns:
        Prompt containing the question and retrieved context.
    """
    return f"""You are answering a question about a software codebase.

Use only the information contained in the provided context.
Do not invent facts that are not supported by the context.
If the context does not contain enough information to answer,
say that the information is not available in the retrieved sources.

Context:
{context}

Question:
{question}

Answer:"""
