import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup

from summarization.html_parsers.parser_base import ParserBase


class ParserBaseTest(unittest.TestCase):

    @classmethod
    @patch("summarization.html_parsers.parser_base.ParserBase.__abstractmethods__", set())
    def setUpClass(cls) -> None:
        cls.parser = ParserBase()

    def test_get_text(self):
        html = """
        <p>
        Text1.<br/>New line.<a>Link</a>
        </p>
        <p>
        Text2.
        </p>
        """
        soup = BeautifulSoup(html, 'html.parser')

        text = self.parser.get_text(soup)

        expected = 'Text1.\nNew line.Link\n\nText2.'
        self.assertEqual(text, expected)
