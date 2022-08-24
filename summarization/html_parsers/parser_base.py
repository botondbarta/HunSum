from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Set, List

import bs4
import pypandoc
from bs4 import BeautifulSoup, Tag

from summarization.models.article import Article
from summarization.models.page import Page


class ParserBase(ABC):
    def __init__(self):
        pass

    def get_article(self, page: Page) -> Article:
        soup = BeautifulSoup(page.html, 'html.parser')
        self.check_page_is_valid(page.url, soup)
        html_soup = self.remove_captions(soup)
        title = self.get_title(page.url, html_soup)
        lead = self.get_lead(html_soup)
        article = self.get_article_text(page.url, html_soup)
        date_of_creation = self.get_date_of_creation(html_soup)
        tags = self.get_tags(html_soup)

        # check if article contains lead
        if article.startswith(lead):
            article = article[len(lead):]

        return Article(title, lead, article, page.domain, page.url, date_of_creation, page.date, list(tags))

    def get_text(self, tag: bs4.Tag, default=None):
        if tag:
            text = pypandoc.convert_text(str(tag), 'plain', format='html', extra_args=['--wrap=none']).strip()
            return text.replace('[]', '')
        return default

    def check_page_is_valid(self, url, soup):
        return

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
    def get_date_of_creation(self, soup) -> Optional[datetime]:
        raise NotImplementedError

    @abstractmethod
    def get_tags(self, soup) -> Set[str]:
        raise NotImplementedError

    @abstractmethod
    def get_html_tags_to_remove(self, soup) -> List[Tag]:
        raise NotImplementedError

    def remove_captions(self, soup) -> BeautifulSoup:
        tags = self.get_html_tags_to_remove(soup)
        for tag in tags:
            tag.decompose()
        return soup
