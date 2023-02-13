from datetime import datetime
from typing import Optional, Set, List

from bs4 import Tag

from summarization.errors.invalid_page_error import InvalidPageError
from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title
from summarization.utils.dateparser import DateParser


class NepszavaParser(ParserBase):

    def check_page_is_valid(self, url, soup):
        if 'kvíz' in [tag.text.lower() for tag in soup.select('div.article-tags > a')]:
            raise InvalidPageError(url, 'Quiz')

    def get_title(self, url: str, soup) -> str:
        title = soup.find('h1', id='article-title')

        if not title:
            title = soup.find('h1', class_='article--title')

        if not title:
            title = soup.find('h1', class_='statseeme')

        if not title:
            titles = soup.find_all('h1')
            tab_title = soup.title
            title = next((x for x in titles if self.get_text(x) in self.get_text(tab_title)), None)

        if not title:
            title = soup.find('h1', itemprop='title')

        if not title:
            title = next(iter(soup.select('div.supertitle')), None)

        assert_has_title(title, url)
        return self.get_text(title)

    def get_lead(self, soup) -> Optional[str]:
        lead = next(iter(soup.select('div.article-lead > p')), None)

        if not lead:
            lead = next(iter(soup.select('div.article--lead')), None)

        if not lead:
            lead = soup.find('div', class_='article-lead')

        if not lead:
            lead = soup.find('h2', itemprop='lead')

        if not lead:
            lead = soup.select_one('div.p.strong')

        if not lead:
            lead = next(iter(soup.select('span.big_text')), None)

        if not lead:
            lead_find = soup.find('meta', property='og:description')
            lead = lead_find["content"] if lead_find else None

        return self.get_text(lead, '')

    def get_article_text(self, url, soup) -> str:
        lead = next(iter(soup.select("div#article-content > div.article-lead")), None)
        if lead:
            lead.decompose()
        article_text = '\n'.join(
            [self.get_text(article, remove_img=True) for article in soup.select('div#article-content')])

        if not article_text:
            content = next(iter(soup.select("div#article-content > div.article-lead")), None)
            if content:
                content.decompose()
            content = next(iter(soup.select("div#article-content > div.article-pager")), None)
            if content:
                content.decompose()
            article_text = self.get_text(next(iter(soup.select('div#article-content')), None), remove_img=True)

        if not article_text:
            article_text = self.get_text(soup.find('section', class_='article--left'))

        if not article_text:
            main_article = soup.find('div', class_='article')
            if main_article:
                article_text = self.get_text(main_article.find('section', class_='left'))

        if not article_text:
            main_article = soup.find('div', class_='article')
            if main_article:
                article_text = '\n'.join(
                    [self.get_text(article, remove_img=True) for article in main_article.select('div.p')])

        if not article_text:
            article_text = self.get_text(soup.find('span', id='text'), remove_img=True)

        if not article_text:
            contents_to_drop = []
            contents_to_drop.extend(soup.findAll('p', itemprop='tags'))
            contents_to_drop.extend(soup.findAll('p', itemprop='date'))
            contents_to_drop.extend(soup.findAll('h1', itemprop='title'))
            contents_to_drop.extend(soup.findAll('h2', itemprop='lead'))
            contents_to_drop.extend(soup.findAll('p', itemprop='author'))
            for content in contents_to_drop:
                content.decompose()
            article_text = soup.select_one('div#forbot > div')

        assert_has_article(article_text, url)
        return article_text

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        date = soup.find('span', id='article-date')

        if not date:
            date = soup.select("p.publish-date > span")

        if not date:
            date = soup.find('p', itemprop='date')

        if not date:
            date = soup.find('div', class_='ctr_left')

        if not date:
            date = soup.find('div', class_='timestamp')
            if date:
                next(iter(soup.select("div.timestamp > div")), None).decompose()

        if not date:
            date = soup.find('div', class_='single-article__news-content__timestamp--date')

        return DateParser.parse(self.get_text(date, ''))

    def get_tags(self, soup) -> Set[str]:
        tags = [self.get_text(tag) for tag in soup.select('div.article-tags > a')]

        if not tags:
            tags = [self.get_text(tag) for tag in soup.select('div.tags > a')]

        if not tags:
            tags_string = self.get_text(soup.find('p', itemprop='tags'))
            if tags_string:
                tags_string = tags_string[:-1]
                tags = tags_string.split(";")

        return set(tags)

    def get_html_tags_to_remove(self, soup) -> List[Tag]:
        to_remove = []
        to_remove.extend(soup.find_all('iframe'))
        to_remove.extend(soup.find_all('div', class_='swiper-slide'))
        to_remove.extend(soup.find_all('div', class_='swiper-slide-active'))
        to_remove.extend(soup.find_all('div', class_='ad'))
        to_remove.extend(soup.select('div#content > div.gallery'))
        to_remove.extend(soup.select('div.goAdverticum'))
        to_remove.extend(soup.select('div.inset-image-container'))
        to_remove.extend(soup.select('script'))
        to_remove.extend(soup.select('link'))
        to_remove.extend(soup.select('a.slogo'))
        to_remove.extend(soup.select('div.images'))
        to_remove.extend(soup.select('div.card'))
        return to_remove
