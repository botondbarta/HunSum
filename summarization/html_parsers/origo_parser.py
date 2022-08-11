import copy
from datetime import datetime
from typing import Optional, Set

import dateparser
from bs4 import BeautifulSoup, Tag

from summarization.errors.invalid_page_error import InvalidPageError
from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title


class OrigoParser(ParserBase):
    def check_page_is_valid(self, url, soup):
        if soup.select('body.gallery') or soup.select('body.page-gallery'):
            raise InvalidPageError(url, 'Gallery')

    def get_title(self, url, soup) -> str:
        title = soup.find('h1', class_='article-title')

        if not title:
            title = soup.find('h1', class_='cikk-cim')

        if not title:
            title = soup.find('div', class_='title')

        if not title:
            title = soup.find('span', class_='cim')

        if not title:
            title = next(iter(soup.select('header#article-head > h1')), None)

        if not title:
            title = next(iter(soup.select('div#article-head > h1')), None)

        if not title:
            title = next(iter(soup.select('div.article_head > h1')), None)

        assert_has_title(title, url)
        return title.get_text(' ').strip()

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('div', id='leades')

        if not lead:
            lead = soup.find('div', class_='article-lead')

        if not lead:
            lead = soup.find('td', id='lead')

        return "" if lead is None else lead.get_text(' ').strip()

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', id='kenyer-szov')

        if not article:
            article = soup.find('td', id='cikktest')

        if not article:
            article = soup.find('div', class_='article-content')

        if not article:
            article = copy.copy(soup.find('div', id='article-text'))
            if article and article.select('div.article-lead'):
                article.select('div.article-lead')[0].decompose()

        if not article:
            article = copy.copy(soup.find('article', id='article-text'))
            if article and article.select('div.article-lead'):
                article.select('div.article-lead')[0].decompose()

        assert_has_article(article, url)
        return article.get_text(' ').strip()

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        date = soup.find('span', class_='cikk-datum')

        if not date:
            date = soup.find('div', class_='date')

        if not date:
            date = soup.find('td', class_='cikkdatum')
            if date:
                return dateparser.parse(date.get_text(' ').split('|')[0].strip())

        if not date:
            date = soup.find('div', class_='article-date')

        if not date:
            date = soup.find('span', id='article-date')

        if not date:
            return None

        return dateparser.parse(date.get_text(' '))

    def get_tags(self, soup) -> Set[str]:
        tags = soup.select('div.article-tags a')
        if tags:
            return set([a.get_text('') for a in tags])

        tag_kozep = soup.select('div#kozep > a')
        if tag_kozep:
            return set([tag.get_text('').strip() for tag in tag_kozep])

        kapcs_cimke = soup.select('div#kapcs-cimke > a')
        if kapcs_cimke:
            return set([tag.get_text('').strip() for tag in kapcs_cimke])

        rovat_cimke = soup.select('div#rovatcimkek > a')
        if rovat_cimke:
            return set([tag.get_text('').strip() for tag in rovat_cimke])

        meta = soup.select('div.article-meta > a')
        if meta:
            return set([tag.get_text('').strip() for tag in meta])

        return set()

    def remove_captions(self, soup) -> BeautifulSoup:
        to_remove = []
        to_remove.extend(soup.find_all('div', id='multidoboz'))
        to_remove.extend(soup.find_all('embedobject', type='lifehukapcscikk'))
        # instagram and maybe more
        to_remove.extend(soup.find_all('div', class_='cikk-obj'))
        to_remove.extend(soup.find_all('div', id='article-related'))
        to_remove.extend(soup.find_all('div', class_='img-holder'))
        to_remove.extend(soup.find_all('figure', class_='newpic'))
        # pics
        to_remove.extend([img.parent for img in soup.select('p[align=center] img')])
        # explanatory texts at the end of the articles
        to_remove.extend(soup.find_all('table', class_='szerk-table'))
        to_remove.extend(soup.find_all('span', class_='source'))

        for r in to_remove:
            r.decompose()
        return soup
