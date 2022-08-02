from typing import Set, Optional
import copy

from bs4 import BeautifulSoup

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_title, assert_has_article


class HvgParser(ParserBase):
    def get_title(self, url, soup) -> str:
        title = soup.find('div', class_='article-title')
        if title is None:
            title = soup.find('h1')
        if title is None:
            a = 2
        assert_has_title(title, url)
        return title.text.strip()

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('div', class_='entry-summary')
        if lead is None:
            leads = soup.select('div.articlecontent > p > strong')
            if leads:
                lead = leads[0]

        return "" if lead is None else lead.text.strip()

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', class_='entry-content')
        if article is None:
            article = next(iter(copy.copy(soup.select('div.articlecontent'))), None)
            if article is not None:
                leads = copy.copy(article.select(' p > strong'))
                if leads:
                    leads[0].decompose()
        if article is None:
            a = 2
        assert_has_article(article, url)
        return article.text.strip()

    def get_tags(self, soup) -> Set[str]:
        tags = soup.select('div.article-tags > a')
        if not tags:
            tags_string = soup.find('b', string='Címkék:')
            if tags_string:
                tags_parent = tags_string.parent
                tags = [tag for tag in tags_parent.children if tag.name == 'a']
                if not tags:
                    tag_div = soup.find('div', class_='location')
                    if tag_div and tag_div.text.lowercase().contains('hvg.hu'):
                        tags = [t for t in tag_div.children if t.name == 'a']
        return set(tag.text for tag in tags)

    def remove_captions(self, soup) -> BeautifulSoup:
        to_remove = []
        to_remove.extend(soup.find_all('figure', class_='article-img'))
        to_remove.extend(soup.find_all('div', class_='video-container'))
        to_remove.extend(soup.find_all('blockquote', class_='twitter-tweet'))
        to_remove.extend(soup.find_all('blockquote', class_='tiktok-embed'))
        to_remove.extend(soup.find_all('div', class_='embedly-card'))
        to_remove.extend(soup.find_all('table', class_='picture'))

        for r in to_remove:
            r.decompose()
        return soup

