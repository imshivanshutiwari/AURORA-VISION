"""Tests for generation/ modules: PromptBuilder, CaptionGenerator, QAGenerator, SummaryGenerator."""
import os
import unittest


class TestPromptBuilder(unittest.TestCase):
    """Tests for PromptBuilder multi-modal prompt construction."""

    def setUp(self):
        from generation.prompt_builder import PromptBuilder

        self.builder = PromptBuilder()

    def test_build_qa_prompt_contains_query(self):
        """build_qa_prompt includes the user's query in the output."""
        prompt = self.builder.build_qa_prompt(
            query="What is happening in this scene?",
            transcript="A person is walking in a park.",
        )
        self.assertIn("What is happening in this scene?", prompt)

    def test_build_qa_prompt_includes_transcript(self):
        """build_qa_prompt includes transcript section when provided."""
        prompt = self.builder.build_qa_prompt(
            query="Who is speaking?",
            transcript="Welcome to AURORA-VISION.",
        )
        self.assertIn("Welcome to AURORA-VISION", prompt)

    def test_build_qa_prompt_includes_ocr_text(self):
        """build_qa_prompt includes on-screen text when provided."""
        prompt = self.builder.build_qa_prompt(
            query="What text is shown?",
            ocr_texts=["BREAKING NEWS", "WEATHER UPDATE"],
        )
        self.assertIn("BREAKING NEWS", prompt)

    def test_build_qa_prompt_includes_system_prompt(self):
        """build_qa_prompt always includes the SYSTEM_PROMPT header."""
        prompt = self.builder.build_qa_prompt(query="test")
        self.assertIn("AURORA-VISION", prompt)


class TestCaptionGeneratorNoLLM(unittest.TestCase):
    """Tests for CaptionGenerator without OpenAI API key."""

    def setUp(self):
        os.environ.pop("OPENAI_API_KEY", None)
        from generation.caption_generator import CaptionGenerator

        self.gen = CaptionGenerator()

    def test_generate_frame_caption_without_llm(self):
        """CaptionGenerator.generate_frame_caption returns Caption even without LLM."""
        from generation.caption_generator import Caption

        caption = self.gen.generate_frame_caption(
            clip_similarity_scores={"person walking": 0.82, "park": 0.74},
            ocr_text=["Park Hours: 6am-10pm"],
            transcript_segment="Welcome to the park.",
        )
        self.assertIsInstance(caption, Caption)
        self.assertIsInstance(caption.text, str)
        self.assertGreater(len(caption.text), 0)


class TestQAPairDataclass(unittest.TestCase):
    """Tests for QAPair dataclass."""

    def test_qa_pair_fields(self):
        """QAPair stores question, answer, confidence, and modalities."""
        from generation.qa_generator import QAPair

        pair = QAPair(
            question="What is shown?",
            answer="A person walking in a park.",
            confidence=0.88,
            source_modalities=["visual", "audio"],
            timestamp_references=[1.5, 3.2],
        )
        self.assertEqual(pair.question, "What is shown?")
        self.assertEqual(pair.answer, "A person walking in a park.")
        self.assertAlmostEqual(pair.confidence, 0.88)
        self.assertIn("visual", pair.source_modalities)
        self.assertEqual(len(pair.timestamp_references), 2)


if __name__ == "__main__":
    unittest.main()
