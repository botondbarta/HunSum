import copy
from datetime import datetime
from typing import Optional, Set, List

from bs4 import Tag
from bs4.element import Comment

from summarization.errors.invalid_page_error import InvalidPageError
from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title
from summarization.utils.dateparser import DateParser


class HvgParser(ParserBase):
    def check_page_is_valid(self, url, soup):
        if soup.select('section.article-pp'):
            raise InvalidPageError(url, 'Liveblog')

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        # new date
        date_tag = soup.find('time', class_='article-datetime')
        date = DateParser.parse(self.get_text(date_tag, ''))
        if date:
            return date

        # older date
        date_tags = soup.select('article.article > p.time > time')
        date = DateParser.parse(self.get_text(date_tags[0], ''))
        if date:
            return date

        # older date
        titles = soup.find_all('h1')
        tab_title = soup.title
        title = next((x for x in titles if self.get_text(x) in self.get_text(tab_title)), None)
        if title:
            p = title.find_next_sibling('p')
            date = DateParser.parse(self.get_text(p).split('\n')[0])
            if date:
                return date

        # oldest date
        time_img = soup.find('img', alt='idő')
        if time_img:
            parent = time_img.parent
            if parent.name == 'a':
                date = DateParser.parse(self.get_text(parent))
                if date:
                    return date

        # gallery date
        divs = soup.find_all('div', class_='fl')
        for div in divs:
            date_string = self.get_text(div).split('\n')[0]
            date = DateParser.parse(date_string)
            if date:
                return date
        return None

    def get_title(self, url, soup) -> str:
        title = soup.find('div', class_='article-title')

        if title is None:
            title = soup.find('h1', class_='articleCaption')

        if title is None:
            titles = soup.find_all('h1')
            tab_title = soup.title
            title = next((x for x in titles if self.get_text(x) in self.get_text(tab_title)), None)

        assert_has_title(title, url)
        return self.get_text(title)

    def get_lead(self, soup) -> Optional[str]:
        # new css
        lead = soup.find('div', class_='entry-summary')

        # older css
        if not lead:
            leads = soup.select('div.articlecontent > p > strong')
            if leads:
                lead = leads[0]

        # old css
        if not lead:
            article_tag = soup.find('article', class_='article')
            lead_comment = [child for child in article_tag.children if
                            isinstance(child, Comment) and 'lead' in child.string]
            if lead_comment:
                lead_p = lead_comment[0].find_next_sibling('p')
                lead = lead_p.next

        return self.get_text(lead, '')

    def get_article_text(self, url, soup) -> str:
        # new css
        article = soup.find('div', class_='entry-content')
        article_text = self.get_text(article, remove_img=True)

        # older css
        if not article or not self.get_text(article):
            article = next(iter(copy.copy(soup.select('div.articlecontent'))), None)
            if article:
                leads = copy.copy(article.select('p > strong'))
                if leads:
                    leads[0].decompose()
            article_text = self.get_text(article, remove_img=True)

        if not article_text:
            columns = soup.find_all('div', class_='columnarticle')
            article_text = ''.join([self.get_text(c, remove_img=True) for c in columns])

        if not article_text:
            article_p = map(lambda a: self.get_text(a), soup.select('article.article > div > p'))
            article_text = '\n'.join(article_p)

        assert_has_article(article_text, url)
        
        return article_text.strip()

    def remove_unnecessary_text_from_article(self, article) -> str:
        article = article.replace('Regisztrálj a Jobline-on, hogy megtaláld álmaid állását és első kézből értesülhess a legújabb munkaerőpiaci trendekről!', '')
        article = article.replace('Kövess minket a Facebook-on is, ahol mindig friss cikkekkel információkkal várunk!', '')
        article = article.replace('Kövess minket a Facebookon!', '')
        article = article.replace('Ide kattintva eléri a Nyüzsi további cikkeit, azonnali véleményeket, érdekességeket, szórakoztató mémeket, gifeket, videókat.', '')
        article = article.replace('Még több Élet + Stílus a Facebook-oldalunkon, kövessen minket:', '')
        return article

    def get_tags(self, soup) -> Set[str]:
        # new css
        tags = soup.select('div.article-tags > a')

        # older css
        if not tags:
            tags_string = soup.find('b', string='Címkék:')
            if tags_string:
                tags_parent = tags_string.parent
                tags = [tag for tag in tags_parent.children if tag.name == 'a']

        # old css
        if not tags:
            tag_div = soup.find('div', class_='location')
            if tag_div and self.get_text(tag_div).lowercase().contains('hvg.hu'):
                tags = [t for t in tag_div.children if t.name == 'a']

        # old css
        if not tags:
            tags = soup.select('article.article > p.tags > a')

        return set(self.get_text(tag) for tag in tags)

    def get_html_tags_to_remove(self, soup) -> List[Tag]:
        to_remove = []
        to_remove.extend(soup.find_all('figure', class_='article-img'))
        to_remove.extend(soup.find_all('div', class_='video-container'))
        to_remove.extend(soup.find_all('blockquote', class_='twitter-tweet'))
        to_remove.extend(soup.find_all('blockquote', class_='tiktok-embed'))
        to_remove.extend(soup.find_all('div', class_='embedly-card'))

        to_remove.extend(soup.find_all('div', class_='embed-container'))

        return to_remove
