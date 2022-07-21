from typing import Set, Optional

from bs4 import BeautifulSoup

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title


class TelexParser(ParserBase):
    def get_title(self, url: str, soup) -> str:
        # new css class
        title = soup.find('div', class_="title-section__top")
        if title is None:
            # old css class
            title = soup.find('div', class_="title-section")
            title = title.h1 if title is not None else None
        assert_has_title(title, url)
        return title.text.strip()

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('p', class_="article__lead")
        return "" if lead is None else lead.text.strip()

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', class_="article-html-content")
        assert_has_article(article, url)
        return article.text

    def get_tags(self, soup) -> Set[str]:
        tags1 = soup.findAll('a', class_="tag--meta")
        tags2 = soup.findAll('meta', {"name": "article:tag"})
        tags3 = soup.findAll('a', class_="meta tag")
        return set(map(lambda t: t.text.strip(), tags1 + tags2 + tags3))

    def remove_captions(self, soup) -> BeautifulSoup:
        soup.find('div', class_='long-img').decompose()
        return soup



