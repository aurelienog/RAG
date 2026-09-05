from transformers import AutoModelForCausalLM, AutoTokenizer

from ..domain import GenerationError
from .prompt import build_prompt


DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_NEW_TOKENS = 512


class AnswerGenerator:
    """Generate grounded answers using Qwen3-0.6B."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
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
        """Generate an answer from the retrieved context."""

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
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

            generated_tokens = outputs[0][inputs.shape[-1]:]

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