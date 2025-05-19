from unittest import TestCase

from tokenizer import Tokenizer


class TokenizerTests(TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()


    def test_encode_decode(self):
        s = "Hello, world!"
        encoded = self.tokenizer.encode(s)
        self.assertEqual(s, self.tokenizer.decode(encoded))

    def test_bos_token(self):
        self.assertEqual(
            self.tokenizer.special_tokens["<|begin_of_text|>"],
            self.tokenizer.bos_token_id
        )

    def test_special_tokens(self):
        self.assertTrue(
            self.tokenizer.bos_token_id in self.tokenizer.special_tokens.values()
        )

