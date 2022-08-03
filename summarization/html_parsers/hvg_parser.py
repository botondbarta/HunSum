from datetime import datetime
from typing import Set, Optional
import dateparser
import copy

from bs4 import BeautifulSoup
from bs4.element import Comment

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_title, assert_has_article


class HvgParser(ParserBase):
    def get_date_of_creation(self, soup) -> datetime:
        # new date
        date_tag = soup.find('time', class_='article-datetime')
        date = dateparser.parse(date_tag.text if date_tag else "")
        if date:
            return date

        # older date
        date_tags = soup.select('article.article > p.time > time')
        date = dateparser.parse(date_tags[0].text if date_tag else "")
        if date:
            return date

        # older date
        titles = soup.find_all('h1')
        tab_title = soup.title
        title = next((x for x in titles if x.text in tab_title.text), None)
        if title:
            p = title.find_next_sibling('p')
            date = dateparser.parse(p.text.strip().split('\n')[0].strip())
            if date:
                return date

        # oldest date
        time_img = soup.find('img', alt='idő')
        if time_img:
            parent = time_img.parent
            if parent.name == 'a':
                date = dateparser.parse(parent.text)
                if date:
                    return date

        # gallery date
        divs = soup.find_all('div', class_='fl')
        for div in divs:
            date_string = div.text.strip().split('\n')[0]
            date = dateparser.parse(date_string)
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
            title = next((x for x in titles if x.text in tab_title.text), None)

        assert_has_title(title, url)
        return title.text.strip()

    def get_lead(self, soup) -> Optional[str]:
        # new css
        lead = soup.find('div', class_='entry-summary')
        if lead:
            return lead.get_text(' ').strip()

        # older css
        leads = soup.select('div.articlecontent > p > strong')
        if leads:
            lead = leads[0]
            if lead:
                return lead.get_text(' ').strip()

        # old css
        article_tag = soup.find('article', class_='article')
        lead_comment = [child for child in article_tag.children if isinstance(child, Comment) and 'lead' in child.string]
        if lead_comment:
            lead_p = lead_comment[0].find_next_sibling('p')
            lead = lead_p.next
            if lead:
                return lead.get_text(' ').strip()

        return ""

    def get_article_text(self, url, soup) -> str:
        # new css
        article = soup.find('div', class_='entry-content')
        if article and article.get_text(" ").strip():
            return article.get_text(' ').strip()

        # older css
        article = next(iter(copy.copy(soup.select('div.articlecontent'))), None)
        if article:
            leads = copy.copy(article.select(' p > strong'))
            if leads:
                leads[0].decompose()
            if article.get_text(" ").strip() == '':
                columns = soup.find_all('div', class_='columnarticle')
                text = ''.join([c.get_text(" ").strip() for c in columns])
                if text != '':
                    return text

        if not article or not article.get_text(' ').strip():
            article_p = map(lambda a: a.get_text(' '), soup.select('article.article > div > p'))
            article_text = '\n'.join(article_p)
            if article_text != '':
                return article_text

        assert_has_article(article, url)
        return article.get_text(' ').strip()

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
            if tag_div and tag_div.get_text(" ").lowercase().contains('hvg.hu'):
                tags = [t for t in tag_div.children if t.name == 'a']

        # old css
        if not tags:
            tags = soup.select('article.article > p.tags > a')

        return set(tag.get_text(" ") for tag in tags)

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




