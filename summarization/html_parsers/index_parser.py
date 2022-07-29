from typing import Set, Optional

from bs4 import BeautifulSoup

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_title, assert_has_article


class IndexParser(ParserBase):
    def get_title(self, url, soup) -> str:
        title = soup.find('div', class_="content-title")
        assert_has_title(title, url)
        return title.text.strip()

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('div', class_="lead")
        return "" if lead is None else lead.text.strip()

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', class_="cikk-torzs")
        assert_has_article(article, url)
        return article.text.strip()

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
        to_remove.extend(soup.find_all('blockquote', class_='tiktok-embed'))
        for r in to_remove:
            r.decompose()
        return soup

