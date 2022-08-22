from datetime import datetime
from typing import Optional, Set

from bs4 import BeautifulSoup

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title
from summarization.utils.dateparser import DateParser


class MetropolParser(ParserBase):
    def get_title(self, url: str, soup) -> str:
        title = soup.find('h1', class_="postTitle")
        if title is None:
            # old css class
            title = soup.find('h1', class_="articleTitle")
        assert_has_title(title, url)
        return self.get_text(title)

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('span', class_="lead")
        if lead is None:
            # old css class
            lead = soup.find('div', class_="lead")
        # TODO check if we need replace
        return self.get_text(lead, '').replace('\n', ' ')

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', class_="postContent")
        if article is None:
            # old css class
            article = soup.find('div', class_="story")
        assert_has_article(article, url)
        return self.get_text_with_filter(article)

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        date = soup.find('div', class_='publicationDate')

        if not date:
            date = soup.find('li', class_='date')

        return DateParser.parse(self.get_text(date))

    def get_tags(self, soup) -> Set[str]:
        tags = soup.find('div', class_="tags")
        return set(self.get_text(tags).split('\n')) if tags else set()

    def remove_captions(self, soup) -> BeautifulSoup:
        to_remove = []
        to_remove.extend(soup.find_all('div', class_='wp-caption'))
        to_remove.extend(soup.find_all('div', class_='endless-shared-area'))
        to_remove.extend(soup.find_all('dl', class_='gallery-item'))

        to_remove.extend(soup.find_all('blockquote', class_='instagram-media'))
        to_remove.extend(soup.find_all('blockquote', class_='twitter-tweet'))
        to_remove.extend(soup.find_all('blockquote', class_='tiktok-embed'))
        to_remove.extend(soup.find_all('div', class_='fb-post'))
        for r in to_remove:
            r.decompose()
        return soup
