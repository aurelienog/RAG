from pathlib import Path

from .ingest import JsonStore
from .models import MinimalSource, RagDataset, StudentSearchResults


class Evaluator:

    def evaluate(self, student_path: Path, dataset_path: Path) -> None:
        """Print recall@k for search results against an answered dataset."""
        store = JsonStore()
        student = store.load_search_results(student_path)
        dataset = store._load(dataset_path, RagDataset, "Dataset")
        ground_truth = {
            question.question_id: question
            for question in dataset.rag_questions
            if hasattr(question, "sources")
        }

        scores = {
            k: self._recall_at_k(student, ground_truth, k)
            for k in (1, 3, 5, 10)
        }
        print("Evaluation Results")
        print("=" * 40)
        print(" ".join(f"Recall@{k}: {score:.3f}" for k, score in scores.items()))

    def _recall_at_k(
        self,
        student: StudentSearchResults,
        ground_truth: dict,
        k: int,
    ) -> float:
        """Calculate recall at k using source-range overlap."""
        total = 0.0
        questions = 0
        for result in student.search_results:
            expected = ground_truth.get(result.question_id)
            if expected is None or not expected.sources:
                continue
            retrieved = result.retrieved_sources[:k]
            found = sum(
                any(self._overlaps(source, candidate) for candidate in retrieved)
                for source in expected.sources
            )
            total += found / len(expected.sources)
            questions += 1
        return total / questions if questions else 0.0

    def _overlaps(self, expected: MinimalSource, candidate: MinimalSource) -> bool:
        """Return whether two source ranges meet the subject overlap threshold."""
        if expected.file_path != candidate.file_path:
            return False
        intersection = max(
            0,
            min(expected.last_character_index, candidate.last_character_index)
            - max(expected.first_character_index, candidate.first_character_index),
        )
        union = max(expected.last_character_index, candidate.last_character_index) - min(
            expected.first_character_index,
            candidate.first_character_index,
        )
        return union > 0 and intersection / union >= 0.05
