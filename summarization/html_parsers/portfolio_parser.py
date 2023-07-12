from datetime import datetime
from typing import Optional, Set, List

from bs4 import Tag

from summarization.errors.invalid_page_error import InvalidPageError
from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title
from summarization.utils.dateparser import DateParser


class PortfolioParser(ParserBase):
    def check_page_is_valid(self, url, soup):
        if soup.select('section.paywall'):
            raise InvalidPageError(url, 'Paywall')

    def get_title(self, url: str, soup) -> str:
        title = soup.select(next(iter('div.pfarticle-title > h1')), None)

        if not title:
            title = soup.find('h1', class_="")

        if not title:
            title = soup.select(next(iter('div.cikk > h1')), None)

        if not title:
            title = soup.select(next(iter('div.title-bar > h1')), None)

        assert_has_title(title, url)
        return self.get_text(title)

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('section', class_='pfarticle-section-lead')

        if not lead:
            tag = soup.find('div', class_='smscontent')
            if tag:
                lead = tag.b

        return self.get_text(lead, '')

    def get_article_text(self, url, soup) -> str:
        article = soup.find('section', class_='section-content')
        article_text = self.get_text(article)

        if not article_text:
            precedent = next(iter(soup.select('section > ul.tags')), None)
            if precedent:
                precedent = precedent.parent
                article_text = '\n'.join(
                    reversed([self.get_text(article) for article in precedent.find_previous_siblings('p')]))

        if not article_text:
            content = soup.find('ul', class_='tags')
            if content:
                content.decompose()
            article_text = '\n'.join(
                [self.get_text(article) for article in soup.select("div.pfarticle-section-content")])

        if not article_text:
            article = soup.find('div', class_='smscontent')
            article_text = self.get_text(article)

        assert_has_article(article_text, url)
        return article_text

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        date = next(iter(soup.select('div.overlay-content > time')), None)

        if not date:
            date = next(iter(soup.select('div.pfarticle-title > time')), None)

        if not date:
            date = next(iter(soup.select('div.author-name > time')), None)

        return DateParser.parse(self.get_text(date, ''))

    def get_tags(self, soup) -> Set[str]:
        tags = [self.get_text(tag) for tag in soup.select('section > ul.tags > li')][1:]

        if not tags:
            tags = [self.get_text(tag) for tag in soup.select('span > a')]

        return set(tags)

    def get_html_tags_to_remove(self, soup) -> List[Tag]:
        to_remove = []
        to_remove.extend(soup.find_all('iframe'))
        to_remove.extend(soup.find_all('figure'))
        to_remove.extend(soup.find_all('span', class_='title-bar'))
        to_remove.extend(soup.select('div.smscontent > b'))
        to_remove.extend(soup.select('div.traderhirdetes'))
        to_remove.extend(soup.select('div.smscontent > p'))
        return to_remove
