"""Unit tests for the keyword retriever (stdlib only: python -m unittest)."""

import unittest

from locallens.retriever import (
    split_paragraphs,
    split_sentences,
    keywords,
    score_paragraph,
    rank,
    build_excerpts,
)


class TestSplitParagraphs(unittest.TestCase):
    def test_blank_lines_split(self):
        text = "First paragraph.\n\nSecond paragraph.\n\n\nThird."
        self.assertEqual(split_paragraphs(text),
                         ["First paragraph.", "Second paragraph.", "Third."])

    def test_empty_text(self):
        self.assertEqual(split_paragraphs("   \n\n  "), [])


class TestSplitSentences(unittest.TestCase):
    def test_splits_on_sentence_boundaries(self):
        text = "First sentence. Second sentence! Is this third? Yes."
        self.assertEqual(split_sentences(text),
                         ["First sentence.", "Second sentence!",
                          "Is this third?", "Yes."])

    def test_keeps_abbreviation_numbers(self):
        text = "Pay $4,500 within fifteen (15) days of invoice."
        self.assertEqual(len(split_sentences(text)), 1)


class TestKeywords(unittest.TestCase):
    def test_stopwords_and_short_words_removed(self):
        kws = keywords("What is the notice period for termination?")
        self.assertIn("notice", kws)
        self.assertIn("period", kws)
        self.assertIn("termination", kws)
        self.assertNotIn("what", kws)
        self.assertNotIn("is", kws)
        self.assertNotIn("the", kws)
        self.assertNotIn("for", kws)


class TestRank(unittest.TestCase):
    paras = [
        "The sky is blue and the ocean is vast.",
        "Either party may terminate with thirty days written notice.",
        "Payment of $4,500 is due within fifteen days of invoice.",
    ]

    def test_relevant_paragraph_ranked_first(self):
        ranked = rank(self.paras, "What is the notice period for termination?")
        self.assertEqual(ranked[0][0], 2)  # 1-based index of termination para
        self.assertIn("terminate", ranked[0][1])

    def test_top_k_defaults_to_one(self):
        ranked = rank(self.paras, "payment invoice")
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0][0], 3)

    def test_no_match_falls_back_to_document_order(self):
        ranked = rank(self.paras, "quantum bananas", top_k=2)
        self.assertEqual([i for i, _ in ranked], [1, 2])

    def test_tie_break_prefers_tighter_span(self):
        paras = [
            "The notice was sent. Many unrelated words follow here and there "
            "and everywhere across a very long rambling paragraph about "
            "nothing. Finally the agreement is mentioned at the very end.",
            "Either party may terminate this agreement with written notice.",
        ]
        ranked = rank(paras, "notice agreement")
        # Both match 2 keywords; para 2's matches sit in one sentence.
        self.assertEqual(ranked[0][0], 2)


class TestStem(unittest.TestCase):
    def test_stemming_matches_inflections(self):
        from locallens.retriever import stem
        self.assertEqual(stem("terminating"), "terminat")
        self.assertEqual(stem("invoices"), "invoice")
        self.assertEqual(stem("notice"), "notice")  # too short to strip

    def test_stemmed_keyword_scores(self):
        count, _ = score_paragraph(
            "Either party may terminate this agreement.", ["terminating"])
        self.assertEqual(count, 1)


class TestBuildExcerpts(unittest.TestCase):
    def test_plain_format_no_numbers(self):
        out = build_excerpts([(2, "Hello."), (5, "World.")])
        self.assertEqual(out, "Hello.\n\nWorld.")


if __name__ == "__main__":
    unittest.main()
