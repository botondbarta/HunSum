from datetime import datetime
from typing import Optional, Set, List

from bs4 import Tag

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title
from summarization.utils.dateparser import DateParser


class M4SportParser(ParserBase):
    def get_title(self, url, soup) -> str:
        title = soup.find('h1', class_='hms_article_title_titletext')

        if not title:
            title = soup.find('h1', class_='hms_article_title')

        if not title:
            title = next(iter(soup.select('div.articleHead > div.articleMeta > h1')), None)

        if not title:
            title = soup.find('h2', class_='hms_article_title')

        if not title:
            title = soup.find('div', class_='titletextm4-redesign')

        if not title:
            title = next(iter(soup.select('div.articleTop > h1')), None)

        if not title:
            titles = soup.find_all('h1')
            tab_title = soup.title
            title = next((x for x in titles if self.get_text(x) in self.get_text(tab_title)), None)

        assert_has_title(title, url)
        return self.get_text(title)

    def get_lead(self, soup) -> str:
        lead = soup.find('span', class_='hms_article_lead_text')

        if not lead:
            lead = next(iter(soup.select('div.hms_article_lead_content > span.hms_article_lead_text')), None)

        if not lead:
            leads = soup.select('div.articleContent > strong > p')
            if leads:
                lead = leads[0]

        if not lead:
            lead = next(iter(soup.select('div.article > p > b')), None)

        return self.get_text(lead, '')

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', class_='hms_article_post_content')

        if not article:
            article = soup.find('div', class_='hms_article_content_wrapper')

        if not article:
            article = next(iter(soup.select('div.articleContent')), None)
            if article:
                leads = article.select('strong > p')
                [lead.decompose() for lead in leads]

                banners = article.select('div.hms-banner-wrapper')
                [banner.decompose() for banner in banners]

                pictures = article.select('div.articlePic')
                [picture.decompose() for picture in pictures]

        if not article:
            article = next(iter(soup.select('div.article')), None)
            article = article.find_all('div', limit=3)[-1]
            if article:
                pictures = article.select('div.articlePic')
                [picture.decompose() for picture in pictures]

        article_text = self.get_text(article, remove_img=True)
        assert_has_article(article_text, url)
        return article_text

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        date = soup.find('span', class_='hms_article_post_date')
        date_text = self.get_text(date, '')
        if "| frissítve:" in date_text:
            date_text = date_text.split("| frissítve:")[0].strip()
            return DateParser.parse(date_text)

        if not date:
            date = soup.find('div', class_='artTime')
            date_text = self.get_text(date, '')
            return DateParser.parse(date_text)

        return DateParser.parse(self.get_text(date, ''))

    def get_tags(self, soup) -> Set[str]:
        tags = [self.get_text(tag) for tag in soup.findAll('span', class_="hms_article_cat_element")]

        if not tags:
            tags = [self.get_text(tag) for tag in soup.findAll('span', class_="hms_video_tags")]

        if not tags:
            tags = [self.get_text(tag) for tag in soup.select('p.path > a')]

        return set(tags)

    def get_html_tags_to_remove(self, soup) -> List[Tag]:
        to_remove = []
        to_remove.extend(soup.find_all('div', class_='twitter-tweet'))
        to_remove.extend(soup.find_all('div', class_='articleImage'))
        to_remove.extend(soup.find_all('iframe', class_='instagram-media'))
        to_remove.extend(soup.find_all('blockquote', class_='instagram-media'))
        to_remove.extend(soup.find_all('blockquote', class_='tiktok-embed'))
        to_remove.extend(soup.find_all('blockquote', class_='twitter-tweet'))
        return to_remove
