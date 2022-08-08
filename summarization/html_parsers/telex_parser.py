from datetime import datetime
from typing import Optional, Set

import dateparser
from bs4 import BeautifulSoup

from summarization.errors.invalid_page_error import InvalidPageError
from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title


class TelexParser(ParserBase):

    def check_page_is_valid(self, url, soup):
        if soup.select('div.liveblog'):
            raise InvalidPageError(url, 'Liveblog')

    def get_title(self, url: str, soup) -> str:
        # new css class
        title = soup.find('div', class_="title-section__top")

        if title is None:
            # old css class
            title = soup.find('div', class_="title-section")
            title = title.h1 if title is not None else None

        if not title:
            title = soup.find('h1', class_='article_title')

        if title is None:
            titles = soup.find_all('h1')
            tab_title = soup.title
            title = next((x for x in titles if x.get_text(' ') in tab_title.get_text(' ')), None)

        assert_has_title(title, url)
        return title.get_text(' ').strip()

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('p', class_="article__lead")
        return "" if lead is None else lead.get_text(' ').strip()

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', class_="article-html-content")
        assert_has_article(article, url)
        return article.get_text(' ')

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        date = soup.find('p', class_='history--original')

        return dateparser.parse(date.get_text(' '))

    def get_tags(self, soup) -> Set[str]:
        tags1 = [tag.get_text(' ').strip() for tag in soup.findAll('a', class_="tag--meta")]
        tags2 = [tag["content"].strip() for tag in soup.findAll('meta', {"name": "article:tag"}) if tag["content"]]
        tags3 = [tag.get_text(' ').strip() for tag in soup.findAll('a', class_="meta tag")]
        return set(tags1 + tags2 + tags3)

    def remove_captions(self, soup) -> BeautifulSoup:
        to_remove = []
        to_remove.extend(soup.find_all('div', class_='long-img'))
        to_remove.extend(soup.find_all('table'))
        for r in to_remove:
            r.decompose()
        return soup
