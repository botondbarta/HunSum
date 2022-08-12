from datetime import datetime
from typing import Optional, Set

from bs4 import BeautifulSoup

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title
from summarization.utils.dateparser import DateParser


class NLCParser(ParserBase):
    def get_title(self, url, soup) -> str:
        title = soup.find('h1', class_='o-post__title')
        if title is None:
            # old css class
            title = soup.find('h2', class_='o-post__title')
        assert_has_title(title, url)
        return self.get_text(title)

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('div', class_='o-post__lead')
        return self.get_text(lead, '')

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', class_='u-onlyArticlePages')
        assert_has_article(article, url)
        return self.get_text(article)

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        date = soup.find('div', class_='o-post__date')

        return DateParser.parse(self.get_text(date))

    def get_tags(self, soup) -> Set[str]:
        tags = soup.find('div', class_='single-post-tags')
        if tags:
            return set(t for t in self.get_text(tags).split('\n'))
        return set()

    def remove_captions(self, soup) -> BeautifulSoup:
        to_remove = []
        to_remove.extend(soup.find_all('div', class_='o-post__authorWrap'))
        to_remove.extend(soup.find_all('div', class_='cikkkeptable'))
        to_remove.extend(soup.find_all('table', class_='cikkkeptable'))
        to_remove.extend(soup.find_all('table', class_='tabla_babazzunk'))
        to_remove.extend(soup.find_all('div', class_='banner-container'))
        to_remove.extend(soup.find_all('div', class_='u-sponsoredBottom'))
        to_remove.extend(soup.find_all('div', class_='wp-caption'))
        to_remove.extend(soup.find_all('div', class_='m-relatedWidget'))
        to_remove.extend(soup.find_all('div', class_='o-cegPostCnt'))
        to_remove.extend(soup.find_all('div', class_='m-embed'))
        # pinterest
        to_remove.extend(soup.find_all('blockquote', class_='embedly-card'))
        # drop recipe parts
        to_remove.extend(soup.find_all('div', class_='recipe-wrapper'))
        for r in to_remove:
            r.decompose()
        return soup
