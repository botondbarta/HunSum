from typing import Optional, Set

from bs4 import BeautifulSoup

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title


class MetropolParser(ParserBase):
    def get_title(self, url: str, soup) -> str:
        title = soup.find('h1', class_="postTitle")
        if title is None:
            # old css class
            title = soup.find('h1', class_="articleTitle")
        assert_has_title(title, url)
        return title.text.strip()

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('span', class_="lead")
        if lead is None:
            # old css class
            lead = soup.find('div', class_="lead")
        return "" if lead is None else lead.text.strip().replace('\n', ' ')

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', class_="postContent")
        if article is None:
            # old css class
            article = soup.find('div', class_="story")
            # for stories that only contains pictures
            article = None if article and article.text.strip() == '' else article
        assert_has_article(article, url)
        return article.text.strip()

    def get_tags(self, soup) -> Set[str]:
        tags = soup.find('div', class_="tags")
        return set(tags.text.strip().split('\n')) if tags else set()

    def remove_captions(self, soup) -> BeautifulSoup:
        return soup
