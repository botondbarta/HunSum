from abc import ABC, abstractmethod
from typing import Optional, Set

from bs4 import BeautifulSoup

from summarization.models.page import Page
from summarization.models.article import Article


class ParserBase(ABC):
    def __init__(self):
        pass

    def get_article(self, page: Page) -> Article:
        html_soup = self.remove_captions(BeautifulSoup(page.html, 'html.parser'))
        title = self.get_title(page.url, html_soup)
        lead = self.get_lead(html_soup)
        article = self.get_article_text(page.url, html_soup)
        tags = self.get_tags(html_soup)
        return Article(title, lead, article, page.domain, page.url, page.date, tags)

    @abstractmethod
    def get_title(self, url, soup) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_lead(self, soup) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def get_article_text(self, url, soup) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_tags(self, soup) -> Set[str]:
        raise NotImplementedError

    @abstractmethod
    def remove_captions(self, soup) -> BeautifulSoup:
        raise NotImplementedError

