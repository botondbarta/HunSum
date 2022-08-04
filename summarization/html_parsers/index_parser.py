from datetime import datetime
from typing import Optional, Set

import dateparser
from bs4 import BeautifulSoup

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title


class IndexParser(ParserBase):
    def get_title(self, url, soup) -> str:
        title = soup.find('div', class_="content-title")
        if not title:
            title = soup.find('div', class_="szoveg")
            if title:
                title = title.h1

        if not title:
            title = soup.find('div', class_="content")
            if title:
                title = title.h1

        if not title:
            title = soup.find('h3', class_="title")

        assert_has_title(title, url)
        return title.text.strip()

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('div', class_="lead")
        return "" if lead is None else lead.text.strip()

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', class_="cikk-torzs")
        if not article:
            article = soup.find('div', class_="text")

        if not article:
            article = soup.find('div', class_='post_text')

        if not article:
            article = soup.find('div', class_="szoveg")

        assert_has_article(article, url)
        return article.text.strip()

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        date = soup.find('div', class_='datum')
        if not date:
            date = soup.find('time', class_='m-asd_time_date')

        if not date:
            date = soup.find('span', class_='ido')

        if not date:
            return None

        date = date.text.strip()
        if 'Módosítva' in date:
            date = date.split("Módosítva")[0].strip()
        try:
            return dateparser.parse(date)
        except:
            return None

    def get_tags(self, soup) -> Set[str]:
        tags_ul = soup.find('ul', class_="cikk-cimkek")
        if tags_ul:
            return set(c.text for c in tags_ul.children)
        return set()

    def remove_captions(self, soup) -> BeautifulSoup:
        to_remove = []
        to_remove.extend(soup.find_all('div', class_='cikk-bottom-text-ad'))
        to_remove.extend(soup.find_all('div', class_='szerkfotogallery'))
        to_remove.extend(soup.find_all('div', class_='szerkfotoimage'))

        to_remove.extend(soup.find_all('div', class_='keretes-donate-doboz'))
        to_remove.extend(soup.find_all('aside', class_='m-automatic-file-snippet'))
        to_remove.extend(soup.find_all('div', class_='m-kepkuldes-box'))
        to_remove.extend(soup.find_all('div', class_='nm_supported__wrapper'))
        to_remove.extend(soup.find_all('div', class_='photographer'))

        to_remove.extend(soup.find_all('div', class_='indavideo'))
        to_remove.extend(soup.find_all('div', id='socialbox_facebook'))
        to_remove.extend(soup.find_all('blockquote', class_='twitter-tweet'))
        to_remove.extend(soup.find_all('div', class_='twitter-tweet'))
        to_remove.extend(soup.find_all('blockquote', class_='tiktok-embed'))
        to_remove.extend(soup.find_all('section', class_='connected'))
        to_remove.extend(soup.find_all('div', class_='post_bottom'))
        for r in to_remove:
            r.decompose()
        return soup
