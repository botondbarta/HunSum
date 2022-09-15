from datetime import datetime
from typing import Optional, Set, List

from bs4 import Tag

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title
from summarization.utils.dateparser import DateParser


class NLCParser(ParserBase):
    def get_title(self, url, soup) -> str:
        title = soup.find('h1', class_='o-post__title')
        if not title:
            # old css class
            title = soup.find('h2', class_='o-post__title')

        if not title:
            title = soup.find('h1', class_='amp-wp-title')

        assert_has_title(title, url)
        return self.get_text(title)

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('div', class_='o-post__lead')

        if not lead:
            lead = soup.find('div', class_='amp-wp-lead')

        return self.get_text(lead, '')

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', class_='single-post-container-content')

        if not article:
            article = soup.find('div', class_='amp-wp-post-content')

        article_text = self.get_text(article, remove_img=True)
        assert_has_article(article_text, url)
        return article_text

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        date = soup.find('div', class_='o-post__date')

        if not date:
            author_and_date = soup.find('div', _class='amp-wp-author')
            if author_and_date:
                date_string = self.get_text(author_and_date, '').split('|')[-1]
                return DateParser.parse(date_string)

        return DateParser.parse(self.get_text(date, ''))

    def get_tags(self, soup) -> Set[str]:
        tags = soup.find('div', class_='single-post-tags')
        if tags:
            return set(t for t in self.get_text(tags).split('\n'))

        tags = soup.select('li > a.tag')
        if tags:
            return set(self.get_text(t) for t in tags)

        return set()

    def get_html_tags_to_remove(self, soup) -> List[Tag]:
        to_remove = []
        to_remove.extend(soup.find_all('div', class_='o-post__authorWrap'))
        to_remove.extend(soup.find_all('div', class_='cikkkeptable'))
        to_remove.extend(soup.find_all('table', class_='cikkkeptable'))
        to_remove.extend(soup.find_all('table', class_='tabla_babazzunk'))
        to_remove.extend(soup.find_all('div', class_='banner-container'))
        to_remove.extend(soup.find_all('div', class_='u-sponsoredBottom'))
        to_remove.extend(soup.find_all('div', class_='wp-caption'))
        to_remove.extend(soup.find_all('div', class_='m-relatedWidget'))
        to_remove.extend(soup.find_all('div', class_='related-picture'))
        to_remove.extend(soup.find_all('div', class_='o-cegPostCnt'))
        to_remove.extend(soup.find_all('div', class_='m-embed'))
        # pinterest
        to_remove.extend(soup.find_all('blockquote', class_='embedly-card'))
        # drop recipe parts
        to_remove.extend(soup.find_all('div', class_='recipe-wrapper'))

        to_remove.extend(soup.find_all('iframe', class_='instagram-media'))
        to_remove.extend(soup.find_all('blockquote', class_='instagram-media'))
        to_remove.extend(soup.find_all('div', class_='fb-video'))

        return to_remove
