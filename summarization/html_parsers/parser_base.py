import unicodedata
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Set, List
from uuid import uuid4

import bs4
import pypandoc
from bs4 import BeautifulSoup, Tag

from summarization.models.article import Article
from summarization.models.page import Page


class ParserBase(ABC):
    def __init__(self):
        pass

    def get_article(self, page: Page) -> Article:
        if page.url in ['https://index.hu/nagykep/2015/10/30/prix_pictet_2015_-_disorder/',
                        'https://index.hu/gazdasag/2016/02/16/adosterkep_nav_adotartozok_listaja_nemzeti_adosterkep_maganszemelyek/',
                        'https://index.hu/tudomany/urkutatas/2010/10/22/eleg_viz_van_a_holdon_egy_urbazishoz/',
                        'https://index.hu/info/2019/11/21/lataselmeny_kompromisszumok_nelkul_progressziv_lencsevel/',
                        'https://index.hu/kulfold/2011/02/21/mozgokepen_az_arab_vilag_lazongasa/9/',
                        'https://index.hu/belfold/2019/01/30/abcug_lestyanszky_adam_siklaki_istvan_szocialpszichologia/',
                        'https://index.hu/leszokik/2014/12/17/orvosi_vizsgalat_video/',
                        'https://index.hu/sport/2002/12/11/kezi2002/']:
            i = 5
        else:
            return Article('','','','','','','','',[''])
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

        return Article(uuid=uuid4().__str__(),
                       title=title,
                       lead=lead,
                       article=unicodedata.normalize('NFKD', article),
                       domain=page.domain,
                       url=page.url,
                       date_of_creation=date_of_creation,
                       cc_date=page.date,
                       tags=list(tags))

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
        tags.extend(soup.find_all('table'))
        for tag in tags:
            tag.decompose()
        return soup
