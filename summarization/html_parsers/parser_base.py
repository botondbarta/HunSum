import os
import pathlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Set

import bs4
import pypandoc
from bs4 import BeautifulSoup

from summarization.models.article import Article
from summarization.models.page import Page


class ParserBase(ABC):
    def __init__(self):
        self.filters = []
        self.filters.append(os.path.join(pathlib.Path(__file__).parent.resolve(), 'filters/image_filter.py'))

    def get_article(self, page: Page) -> Article:
        soup = BeautifulSoup(page.html, 'html.parser')
        self.check_page_is_valid(page.url, soup)
        html_soup = self.remove_captions(soup)
        title = self.get_title(page.url, html_soup)
        lead = self.get_lead(html_soup)
        article = self.get_article_text(page.url, html_soup)
        date_of_creation = self.get_date_of_creation(html_soup)
        tags = self.get_tags(html_soup)
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
    def remove_captions(self, soup) -> BeautifulSoup:
        raise NotImplementedError
