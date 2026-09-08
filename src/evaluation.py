from pathlib import Path

from .domain import EvaluationError
from .ingest import JsonStore
from .models import (
    AnsweredQuestion,
    MinimalSource,
    StudentSearchResults,
)


class Evaluator:
    """Evaluate retrieval results against a ground-truth dataset.

    Retrieval performance is measured using recall@k, where a retrieved
    source is considered relevant when its character range overlaps the
    expected source range by at least the configured IoU threshold.

    Attributes:
        EVALUATION_K: Values of k used when calculating recall.
        IOU_THRESHOLD: Minimum intersection-over-union required for two source
            ranges to be considered overlapping.
    """

    EVALUATION_K = (1, 3, 5, 10)
    IOU_THRESHOLD = 0.05

    def evaluate(
        self,
        student_path: Path,
        dataset_path: Path,
    ) -> None:
        """Evaluate and print retrieval recall at predefined k values.

        The student search results are compared against answered questions
        from the ground-truth dataset. Recall is calculated for each value in
        ``EVALUATION_K`` and printed to standard output.

        Args:
            student_path: Path to the student search results file.
            dataset_path: Path to the ground-truth dataset.

        Raises:
            EvaluationError: If the dataset contains no answered questions or
                if no matching answered questions are found between the search
                results and the dataset.
        """

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
        """Calculate average recall@k using source-range overlap.

        A ground-truth source is counted as retrieved when at least one of the
        top-k candidate sources overlaps its character range by at least
        ``IOU_THRESHOLD``.

        Args:
            student: Search results produced by the student implementation.
            ground_truth: Mapping of question identifiers to answered
                questions containing the expected source ranges.
            k: Number of retrieved sources considered for each question.

        Returns:
            The average recall@k across all matching questions that contain
            at least one expected source.

        Raises:
            EvaluationError: If ``k`` is not greater than zero or if no
                matching questions with expected sources are found.
        """

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
        """Check whether two source ranges meet the IoU threshold.

        Sources from different files are never considered overlapping. For
        sources in the same file, the intersection-over-union of their
        character ranges is compared against ``IOU_THRESHOLD``.

        Args:
            expected: Ground-truth source range.
            candidate: Retrieved source range.

        Returns:
            ``True`` if both sources belong to the same file and their
            character ranges have an IoU greater than or equal to
            ``IOU_THRESHOLD``; otherwise, ``False``.
        """

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
