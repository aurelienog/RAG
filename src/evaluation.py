from pathlib import Path

from .domain import EvaluationError
from .ingest import JsonStore
from .models import (
    AnsweredQuestion,
    MinimalSource,
    StudentSearchResults,
)


class Evaluator:
    """Evaluate retrieval results against the ground-truth dataset."""

    EVALUATION_K = (1, 3, 5, 10)
    IOU_THRESHOLD = 0.05

    def evaluate(
        self,
        student_path: Path,
        dataset_path: Path,
    ) -> None:
        """Print recall@k for search results against an answered dataset."""

        store = JsonStore()

        student = store.load_search_results(student_path)
        dataset = store.load_dataset(dataset_path)

        ground_truth: dict[str, AnsweredQuestion] = {
            question.question_id: question
            for question in dataset.rag_questions
            if isinstance(question, AnsweredQuestion)
        }

        if not ground_truth:
            raise EvaluationError(
                "Dataset does not contain answered questions."
            )

        scores = {
            k: self._recall_at_k(
                student,
                ground_truth,
                k,
            )
            for k in self.EVALUATION_K
        }

        print("Evaluation Results")
        print("=" * 40)
        print(
            " ".join(
                f"Recall@{k}: {score:.3f}"
                for k, score in scores.items()
            )
        )

    def _recall_at_k(
        self,
        student: StudentSearchResults,
        ground_truth: dict[str, AnsweredQuestion],
        k: int,
    ) -> float:
        """Calculate recall@k using source-range overlap."""

        if k <= 0:
            raise EvaluationError(
                "k must be greater than zero."
            )

        total = 0.0
        questions = 0

        for result in student.search_results:
            expected = ground_truth.get(result.question_id)

            if expected is None:
                continue

            if not expected.sources:
                continue

            retrieved = result.retrieved_sources[:k]

            found = sum(
                any(
                    self._overlaps(source, candidate)
                    for candidate in retrieved
                )
                for source in expected.sources
            )

            total += found / len(expected.sources)
            questions += 1

        if questions == 0:
            raise EvaluationError(
                "No matching answered questions were found "
                "between search results and dataset."
            )

        return total / questions

    def _overlaps(
        self,
        expected: MinimalSource,
        candidate: MinimalSource,
    ) -> bool:
        """Return whether two source ranges meet the IoU threshold."""

        if expected.file_path != candidate.file_path:
            return False

        intersection = max(
            0,
            min(
                expected.last_character_index,
                candidate.last_character_index,
            )
            - max(
                expected.first_character_index,
                candidate.first_character_index,
            ),
        )

        union = (
            max(
                expected.last_character_index,
                candidate.last_character_index,
            )
            - min(
                expected.first_character_index,
                candidate.first_character_index,
            )
        )

        return (
            union > 0
            and intersection / union >= self.IOU_THRESHOLD
        )
