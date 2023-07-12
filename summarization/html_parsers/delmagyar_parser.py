import copy
from datetime import datetime
from typing import Optional, Set, List

from bs4 import Tag

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title
from summarization.utils.dateparser import DateParser


class DelmagyarParser(ParserBase):

    def get_title(self, url: str, soup) -> str:
        title = soup.find('h2', class_='single-article__title')

        if not title:
            title = next(iter(soup.select('div#article_content h2')), None)

        if not title:
            title = soup.find('h1', class_='article-title_h1', id='article-title_h1')

        if not title:
            title = next(iter(soup.select('article.left-column h1')), None)

        if not title:
            title = soup.find('h1', id='article-title_h1')

        if not title:
            title = soup.find('span', class_='title')

        assert_has_title(title, url)

        title_text = self.get_text(title)
        title_text = title_text.replace('½', '"')
        return title_text

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('div', class_='single-article__lead')

        if not lead:
            lead = soup.find('div', class_='enews-article-lead')

        if not lead:
            lead = soup.find('h1', class_='article-title_h1')
        if not lead:
            lead = soup.find('div', class_='contentLead')

        if not lead:
            lead = soup.find('p', class_='lead')

        if not lead:
            lead = soup.find('h4', id='article_lead')

        return self.get_text(lead, '')

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', class_='article_szoveg')

        if not article:
            article = copy.copy(soup.find('div', class_='enews-article-content'))
            if article:
                tags = article.find('div', class_='tagsContainer')
                if tags:
                    tags.decompose()

        if not article:
            article = soup.find('div', class_='content')

        if not article:
            article = '\n'.join([self.get_text(par) for par in soup.findAll('div', class_='block-content')])

        article_text = self.get_text(article)
        assert_has_article(article_text, url)

        return article_text.strip()

    def remove_unnecessary_text_from_article(self, article):
        article = article.replace('Írásunkat keresse szombaton a Szieszta mellékletben!', '')
        article = article.replace('Fizessen elõ a napilapra!', '')
        return article

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        date = next(iter(soup.select('div.overlay-content > time')), None)

        if not date:
            date = next(iter(soup.select('div.pfarticle-title > time')), None)

        if not date:
            date = next(iter(soup.select('div.author-name > time')), None)

        if not date:
            date = soup.find('span', class_='created')

        if not date:
            date = soup.find('span', class_='datum')

        if not date:
            date = soup.find('span', class_='article-datetime')

        if not date:
            date = soup.find('div', class_='article-meta-box--time')

        if not date:
            date = soup.find('p', class_='time')

        return DateParser.parse(self.get_text(date, ''))

    def get_tags(self, soup) -> Set[str]:
        tags = [self.get_text(tag) for tag in soup.select('div.tag')]

        if not tags:
            tags = [self.get_text(tag) for tag in soup.select('div.single-article__labels > a.label')]

        return set(tags)

    def get_html_tags_to_remove(self, soup) -> List[Tag]:
        to_remove = []
        to_remove.extend(soup.select('script'))
        to_remove.extend(soup.find_all('div', class_='related'))
        to_remove.extend(soup.find_all('figcaption'))
        to_remove.extend(soup.find_all('div', class_='comment_box'))
        to_remove.extend(soup.find_all('div', class_='articleInArticleOfferer'))
        to_remove.extend(soup.find_all('div', class_='comments_button_a'))
        to_remove.extend(soup.find_all('div', class_='endless-shared-area'))
        to_remove.extend(soup.find_all('div', class_='tagsLabel'))
        to_remove.extend(soup.find_all('div', class_='enews-article-offerer-info'))
        to_remove.extend(soup.find_all('div', class_='enews-article-offerer'))
        to_remove.extend(soup.find_all('div', class_='withVideoEmbed'))
        to_remove.extend(soup.find_all('div', class_='et-top-navigation'))
        to_remove.extend(soup.find_all('div', class_='adult-layer'))
        to_remove.extend(soup.find_all('div', id='et-top-navigation'))
        to_remove.extend(soup.find_all('div', id='kapcsolodo_cikk'))
        to_remove.extend(soup.find_all('div', id='article_data_2'))
        to_remove.extend(soup.find_all('header', id='main-header'))
        to_remove.extend(soup.find_all('footer'))
        to_remove.extend(soup.find_all('header', class_='withVideoEmbed'))
        to_remove.extend(soup.find_all('div', class_='portfolio_class'))
        to_remove.extend(soup.find_all('img'))
        return to_remove
