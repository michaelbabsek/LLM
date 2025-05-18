from distutils.core import setup_keywords
from unittest import TestCase

from tokenizer import Tokenizer


class TokenizerTests(TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()


    def test_encode_decode(self):
        s = "Hello, world!"
        encoded = self.tokenizer.encode(s)
        self.assertEqual(s, self.tokenizer.decode(encoded))

