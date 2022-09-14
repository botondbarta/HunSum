import os
import pathlib
import unittest

from bs4 import BeautifulSoup

from summarization.errors.invalid_page_error import InvalidPageError
from summarization.html_parsers.telex_parser import TelexParser
from summarization.utils.dateparser import DateParser


class TelexParserTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = TelexParser()

    def _get_soup(self, filename) -> BeautifulSoup:
        resource_dir = pathlib.Path(__file__).parent.parent.resolve() / 'resources' / 'telex'
        with open(os.path.join(resource_dir, filename), 'r') as f:
            html = f.read()
            return BeautifulSoup(html, 'html.parser')

    def test_get_title_1(self):
        soup = self._get_soup('telex_1.html')

        title = self.parser.get_title('url', soup)

        self.assertEqual(title, 'Title')

    def test_get_title_2(self):
        soup = self._get_soup('telex_2.html')

        title = self.parser.get_title('url', soup)

        self.assertEqual(title, 'Title')

    def test_get_title_3(self):
        soup = self._get_soup('telex_3.html')

        title = self.parser.get_title('url', soup)

        self.assertEqual(title, 'Title')

    def test_get_title_4(self):
        soup = self._get_soup('telex_4.html')

        title = self.parser.get_title('url', soup)

        self.assertEqual(title, 'Title')

    def test_get_lead_1(self):
        soup = self._get_soup('telex_1.html')

        lead = self.parser.get_lead(soup)

        self.assertEqual(lead, 'Lead')

    def test_get_tags_1(self):
        soup = self._get_soup('telex_1.html')

        tags = self.parser.get_tags(soup)

        self.assertEqual(set(tags), {'tag1', 'tag2', 'tag3', 'tag4', 'tag5'})

    def test_get_article_1(self):
        soup = self._get_soup('telex_1.html')

        article = self.parser.get_article_text('url', soup)

        expected = 'Text link text.\nNew line.\n\nNew paragraph.'
        self.assertEqual(article, expected)

    def test_get_date_1(self):
        soup = self._get_soup('telex_1.html')

        date = self.parser.get_date_of_creation(soup)

        self.assertEqual(date, DateParser.parse('2021. január 14. – 10:22'))

    def test_get_date_2(self):
        soup = self._get_soup('telex_2.html')

        date = self.parser.get_date_of_creation(soup)

        self.assertEqual(date, DateParser.parse('2021. január 14. – 10:22'))

    def test_get_date_3(self):
        soup = self._get_soup('telex_3.html')

        date = self.parser.get_date_of_creation(soup)

        self.assertEqual(date, DateParser.parse('2021. január 14. – 10:22'))

    def test_get_html_tags_to_remove(self):
        soup = self._get_soup('telex_4.html')

        html_tags = self.parser.get_html_tags_to_remove(soup)

        self.assertEqual(len(html_tags), 2)

    def test_check_if_page_is_valid(self):
        soup = self._get_soup('telex_4.html')

        with self.assertRaises(InvalidPageError):
            self.parser.check_page_is_valid('url', soup)

