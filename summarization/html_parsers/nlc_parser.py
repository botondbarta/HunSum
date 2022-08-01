from typing import Set, Optional

from bs4 import BeautifulSoup

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_title, assert_has_article


class NLCParser(ParserBase):
    def get_title(self, url, soup) -> str:
        title = soup.find('h1', class_="o-post__title")
        assert_has_title(title, url)
        return title.text.strip()

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('div', class_="o-post__lead")
        return "" if lead is None else lead.text.strip()

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', class_="u-onlyArticlePages")
        assert_has_article(article, url)
        return article.text

    def get_tags(self, soup) -> Set[str]:
        tags_ul = soup.find('ul', class_="cikk-cimkek")
        if tags_ul:
            return set(c.text for c in tags_ul.children)
        return set()

    def remove_captions(self, soup) -> BeautifulSoup:
        to_remove = []
        for r in to_remove:
            r.decompose()
        return soup

