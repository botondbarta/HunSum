from datetime import datetime
from typing import Optional, Set, List

from bs4 import Tag

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title
from summarization.utils.dateparser import DateParser


class KisalfoldParser(ParserBase):
    def get_title(self, url, soup) -> str:
        title = soup.find('h2', class_='single-article__title')

        if not title:
            title = soup.find('h1', class_='article-title_h1')

        if not title:
            title = soup.find('h1', id='article-title_h1')

        if not title:
            title = next(iter(soup.select('div#article_content > h2')), None)

        if not title:
            title = next(iter(soup.select('article.left-column > h1')), None)

        assert_has_title(title, url)
        return self.get_text(title)

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('div', class_='single-article__lead')

        if not lead:
            lead = soup.find('div', class_='enews-article-lead')

        if not lead:
            lead = soup.find('h4', id='article_lead')

        if not lead:
            lead = next(iter(soup.select('article.left-column > p.lead')), None)

        return self.get_text(lead, '')

    def get_article_text(self, url, soup) -> str:
        article_text = self.get_text(soup.find('div', id='article_text'))

        if not article_text:
            article = soup.select('div.enews-article-content > p')
            article_text = "".join([self.get_text(text, remove_img=True) for text in article])

        if not article_text:
            article = soup.select('div.enews-article-content > div')
            article_text = "".join([self.get_text(text, remove_img=True) for text in article])

        if not article_text:
            article = soup.select('div.block-content > p')
            article_text = "".join([self.get_text(text, remove_img=True) for text in article])

        assert_has_article(article_text, url)
        return article_text

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        date = soup.find('span', class_='created')

        if not date:
            date = soup.find('span', id='article-datetime_1')

        if not date:
            date = soup.find('div', class_='article-meta-box--time')

        if not date:
            date = soup.find('span', class_='datum')

        if not date:
            date = soup.find('p', class_='time')

        return DateParser.parse(self.get_text(date, ''))

    def get_tags(self, soup) -> Set[str]:
        tags = [self.get_text(tag) for tag in soup.select('div.single-article__labels > a')]

        if not tags:
            tags = [self.get_text(tag) for tag in soup.select('div.tags > div.tag')]

        if not tags:
            tags = [self.get_text(tag) for tag in soup.select('div.tags > a')]

        if not tags:
            tags = [self.get_text(tag) for tag in soup.select('div.article_labels > a')]

        return set(tags)

    def get_html_tags_to_remove(self, soup) -> List[Tag]:
        to_remove = []
        to_remove.extend(soup.find_all('div', class_='cikk-obj'))
        to_remove.extend(soup.find_all('div', id='article-related'))
        to_remove.extend(soup.find_all('div', class_='img-holder'))
        to_remove.extend(soup.find_all('figure', class_='newpic'))
        to_remove.extend(soup.find_all('span', class_='source'))
        to_remove.extend(soup.select('div#article_text > div#kapcsolodo_cikk'))
        to_remove.extend(soup.select('div#article_text > div#article_data_2'))
        to_remove.extend(soup.select('div#article_text > div.lapcomallas'))
        return to_remove
