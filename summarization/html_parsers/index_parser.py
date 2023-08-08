import copy
from datetime import datetime
from typing import Optional, Set, List

from bs4 import Tag, BeautifulSoup

from summarization.errors.invalid_page_error import InvalidPageError
from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title
from summarization.utils.dateparser import DateParser


class IndexParser(ParserBase):
    def check_page_is_valid(self, url, soup):
        if soup.select('div.pp-article-site'):
            raise InvalidPageError(url, 'Liveblog')

    def get_title(self, url, soup) -> str:
        title = soup.find('div', class_="content-title")

        if not title:
            title = next(iter(soup.select('div.szoveg > h1')), None)

        if not title:
            title = next(iter(soup.select('div.content > h1')), None)

        if not title:
            title = soup.find('h3', class_="title")

        assert_has_title(title, url)
        return self.get_text(title)

    def get_lead(self, soup: BeautifulSoup) -> Optional[str]:
        lead = soup.find('div', class_="lead")

        if not lead:
           lead = soup.find('p', class_='ctl05_lbLead')

        if not lead:
            first_p = soup.select_one('div.cikk-torzs > p')
            lead = first_p.find(
                # the p tag starts with <strong> tag instead of text
                lambda t: t.name == 'strong' and (not t.previous_sibling or str(t.previous_sibling).isspace()))

        return self.get_text(lead, '')

    def get_article_text(self, url, soup) -> str:
        article = copy.copy(soup.find('div', class_="cikk-torzs"))
        # remove lead if exists
        if article:
            first_p = soup.select_one('div.cikk-torzs > p')
            lead = first_p.find(
                # the p tag starts with <strong> tag instead of text
                lambda t: t.name == 'strong' and (not t.previous_sibling or str(t.previous_sibling).isspace()))
            if lead:
                lead.decompose()

            to_decompose = []
            to_decompose += article.select('div.cikk-torzs > div > ul.m-tag-list')
            to_decompose += article.select('div.cikk-torzs > aside')
            to_decompose += article.select('div.cikk-torzs > div.content-disclaimer-text')

            for decomposable in to_decompose:
                decomposable.decompose()

        if not article:
            article = soup.find('div', class_="text")

        if not article:
            article = soup.find('div', class_='post_text')

        if not article:
            article = soup.find('div', class_="szoveg")

        article_text = self.get_text(article, remove_img=True)
        assert_has_article(article_text, url)

        return article_text

    def remove_unnecessary_text_from_article(self, article):
        article = article.replace('Kövesse az Indexet Facebookon is!', '')
        return article

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        date = soup.find('div', class_='datum')
        if not date:
            date = soup.find('time', class_='m-asd_time_date')

        if not date:
            date = soup.find('span', class_='ido')

        date = self.get_text(date, '')
        if 'Módosítva' in date:
            date = date.split("Módosítva")[0].strip()

        return DateParser.parse(date)

    def get_tags(self, soup) -> Set[str]:
        tags = soup.select('ul.cikk-cimkek > li > a')
        return set(self.get_text(tag) for tag in tags)

    def get_html_tags_to_remove(self, soup) -> List[Tag]:
        to_remove = []
        to_remove.extend(soup.find_all('div', class_='cikk-bottom-text-ad'))
        to_remove.extend(soup.find_all('div', class_='cikk-bottom-box'))
        to_remove.extend(soup.find_all('div', class_='pr__cikk-bottomcontent'))
        to_remove.extend(soup.find_all('nav', class_='pager'))
        to_remove.extend(soup.find_all('div', class_='iframe-embed-container'))
        to_remove.extend(soup.find_all('div', class_='szerkfotogallery'))
        to_remove.extend(soup.find_all('div', class_='szerkfotoimage'))

        to_remove.extend(soup.find_all('div', class_='keretes-donate-doboz'))
        to_remove.extend(soup.find_all('aside', class_='m-automatic-file-snippet'))
        to_remove.extend(soup.find_all('div', class_='m-kepkuldes-box'))
        to_remove.extend(soup.find_all('div', class_='nm_supported__wrapper'))
        to_remove.extend(soup.find_all('div', class_='nm_thanks__wrapper'))
        to_remove.extend(soup.find_all('div', class_='photographer'))

        to_remove.extend(soup.find_all('div', class_='indavideo'))
        to_remove.extend(soup.find_all('div', id='socialbox_facebook'))
        to_remove.extend(soup.find_all('div', id='index-social-box'))
        to_remove.extend(soup.find_all('blockquote', class_='twitter-tweet'))
        to_remove.extend(soup.find_all('div', class_='twitter-tweet'))
        to_remove.extend(soup.find_all('blockquote', class_='tiktok-embed'))
        to_remove.extend(soup.find_all('section', class_='connected'))
        to_remove.extend(soup.find_all('div', class_='post_bottom'))
        to_remove.extend(soup.find_all('div', class_='nm_mini__wrapper'))
        to_remove.extend(soup.find_all('div', class_='table_container'))
        to_remove.extend(soup.find_all('div', class_='szoveg_spec_container'))

        to_remove.extend(soup.find_all('div', class_='index_kep_gal_ala'))
        to_remove.extend(soup.find_all('div', class_='hirverseny'))
        to_remove.extend(soup.find_all('p', class_='meta-twitter__copy'))
        to_remove.extend(soup.find_all('div', class_='meta-twitter__btns'))
        to_remove.extend(soup.find_all('div', class_='szelsojobb'))

        return to_remove
