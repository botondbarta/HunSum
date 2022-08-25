from datetime import datetime
from typing import Optional, Set, List

from bs4 import Tag

from summarization.errors.invalid_page_error import InvalidPageError
from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title
from summarization.utils.dateparser import DateParser


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

        if not title:
            titles = soup.find_all('h1')
            tab_title = soup.title
            title = next((x for x in titles if self.get_text(x) in self.get_text(tab_title)), None)

        assert_has_title(title, url)
        return self.get_text(title)

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('p', class_="article__lead")
        return self.get_text(lead, '')

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', class_="article-html-content")
        assert_has_article(article, url)
        return self.get_text(article)

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        date = soup.find('p', class_='history--original')

        if not date:
            date = soup.find('p', id='original_date')

        if not date:
            date = soup.find('div', class_='article_date')
            date_object = DateParser.parse(self.get_text(date, '').split('\n')[0])
            if date_object:
                return date_object

        return DateParser.parse(self.get_text(date, ''))

    def get_tags(self, soup) -> Set[str]:
        tags1 = [self.get_text(tag).lstrip('#') for tag in soup.findAll('a', class_="tag--meta")]
        tags2 = [tag["content"].strip() for tag in soup.findAll('meta', {"name": "article:tag"}) if tag["content"]]
        tags3 = [self.get_text(tag).lstrip('#') for tag in soup.findAll('a', class_="meta tag")]
        return set(tags1 + tags2 + tags3)

    def get_html_tags_to_remove(self, soup) -> List[Tag]:
        to_remove = []
        to_remove.extend(soup.find_all('div', class_='long-img'))
        to_remove.extend(soup.find_all('table'))

        return to_remove
