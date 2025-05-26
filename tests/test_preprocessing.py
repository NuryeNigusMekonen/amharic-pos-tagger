import unittest
from src.preprocess import build_vocab, build_tagset, encode_sentence

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.sentences = [[("እሱ", "PRON"), ("አለ", "VERB")]]
        self.vocab = build_vocab(self.sentences)
        self.tag2idx = build_tagset(self.sentences)

    def test_vocab(self):
        self.assertIn("እሱ", self.vocab)
        self.assertIn("<PAD>", self.vocab)

    def test_tagset(self):
        self.assertEqual(self.tag2idx["PRON"], 1)  # depending on sorting

    def test_encoding(self):
        encoded = encode_sentence(self.sentences[0], self.vocab, self.tag2idx)
        self.assertEqual(len(encoded[0]), 2)
        self.assertEqual(len(encoded[1]), 2)

if __name__ == "__main__":
    unittest.main()