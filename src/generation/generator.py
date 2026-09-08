from transformers import AutoModelForCausalLM, AutoTokenizer

from ..domain import GenerationError
from .prompt import build_prompt


DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_NEW_TOKENS = 512


class AnswerGenerator:
    """Generate grounded answers using the configured language model.

    The generator uses a Hugging Face causal language model and tokenizer
    to produce answers based on a question and retrieved document context.

    Attributes:
        max_new_tokens: Maximum number of new tokens generated for an answer.
        tokenizer: Tokenizer associated with the configured language model.
        model: Configured causal language model used for text generation.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        """Initialize the answer generator and load the language model.

        Args:
            model_name: Name or path of the Hugging Face model to load.
            max_new_tokens: Maximum number of new tokens the model can
                generate for each answer.

        Raises:
            GenerationError: If ``model_name`` is empty, if
                ``max_new_tokens`` is not greater than zero, or if the model
                or tokenizer cannot be loaded.
        """
        if not model_name.strip():
            raise GenerationError(
                "Model name cannot be empty."
            )

        if max_new_tokens <= 0:
            raise GenerationError(
                "max_new_tokens must be greater than zero."
            )

        self.max_new_tokens = max_new_tokens

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise GenerationError(
                f"Could not load model: {model_name}"
            ) from exc

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        """Generate an answer grounded in the provided context.

        Args:
            question: Question that the model should answer.
            context: Retrieved document context used to ground the answer.

        Returns:
            The generated answer as a string. If the context is empty, a
            message indicating that there is insufficient relevant
            information is returned.

        Raises:
            GenerationError: If the question is empty, if text generation
                fails, or if the model produces an empty answer.
        """

        if not question.strip():
            raise GenerationError(
                "Question cannot be empty."
            )

        if not context.strip():
            return (
                "I could not find enough relevant information "
                "in the retrieved sources to answer the question."
            )

        prompt = build_prompt(
            question=question,
            context=context,
        )

        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

            generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

            answer = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

        except (OSError, RuntimeError, ValueError) as exc:
            raise GenerationError(
                "Could not generate an answer."
            ) from exc

        if not answer:
            raise GenerationError(
                "The model returned an empty answer."
            )

        return answer
