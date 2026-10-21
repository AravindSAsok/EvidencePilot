
import importlib.util
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_PATH = PROJECT_ROOT / "pages" / "4_Research_Agent.py"


def load_agent():
    spec = importlib.util.spec_from_file_location(
        "research_agent", AGENT_PATH
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Research Agent from {AGENT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = load_agent()


class TestResearchAgent(unittest.TestCase):

    def test_tokenize_normalizes_text(self):
        result = agent.tokenize("EvidencePilot Research")

        self.assertTrue(result)
        self.assertTrue(
            all(isinstance(token, str) for token in result)
        )

    def test_chunk_pdf_pages_preserves_page_numbers(self):
        pages = [
            {"page": 1, "text": "Research evidence " * 150},
            {"page": 2, "text": "Study findings " * 150},
        ]

        chunks = agent.chunk_pdf_pages(pages)

        self.assertTrue(chunks)
        self.assertTrue(
            all(isinstance(chunk, dict) for chunk in chunks)
        )
        self.assertTrue(
            all("page" in chunk for chunk in chunks)
        )

    def test_keyword_ranking_returns_results(self):
        chunks = [
            {
                "id": "chunk-1",
                "page": 1,
                "text": "The study examines climate change.",
            },
            {
                "id": "chunk-2",
                "page": 2,
                "text": "The paper discusses medical research.",
            },
        ]

        results = agent.rank_keyword("climate change", chunks)

        self.assertIsInstance(results, list)
        self.assertTrue(
            all(isinstance(item, dict) for item in results)
        )

    def test_hybrid_ranking_returns_results(self):
        keyword_results = [
            {
                "chunk_id": "chunk-1",
                "page": 1,
                "text": "Climate research",
                "score": 2.0,
            }
        ]

        semantic_results = [
            {
                "chunk_id": "chunk-2",
                "page": 2,
                "text": "Climate study",
                "score": 0.9,
            }
        ]

        results = agent.rank_hybrid(
            keyword_results, semantic_results
        )

        self.assertIsInstance(results, list)
        self.assertTrue(
            all(isinstance(item, dict) for item in results)
        )

    def test_retrieval_metrics_return_expected_fields(self):
        ranking = [
            {"page": 2, "text": "Expected evidence"},
            {"page": 1, "text": "Other evidence"},
        ]

        metrics = agent.calculate_retrieval_metrics(
            ranking, {2}
        )

        self.assertIsInstance(metrics, dict)
        self.assertTrue(metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
