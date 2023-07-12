from datetime import datetime
from typing import Optional, Set, List

from bs4 import Tag

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.assertion import assert_has_article, assert_has_title
from summarization.utils.dateparser import DateParser


class Parser24(ParserBase):
    def get_title(self, url, soup) -> str:
        title = soup.find('h1', class_='o-post__title')

        if not title:
            # old css class
            title = soup.find('h1', class_='post-title')

        if not title:
            title = soup.find('h1', class_='amp-wp-title')

        assert_has_title(title, url)
        return self.get_text(title)

    def get_lead(self, soup) -> Optional[str]:
        lead = soup.find('div', class_='lead')

        if not lead:
            lead = soup.find('div', class_='amp-wp-lead')

        return self.get_text(lead, '')

    def get_article_text(self, url, soup) -> str:
        article = soup.find('div', class_='post-body')

        if not article:
            article = soup.find('div', class_='amp-wp-post-content')

        article_text = self.get_text(article, remove_img=True)
        assert_has_article(article_text, url)

        return article_text

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        date = soup.find('div', class_='author-content')
        if date and date.p:
            return DateParser.parse(self.get_text(date.p))

        date = soup.find('span', class_='o-post__date')
        if date:
            date_text = self.get_text(date)
            if 'FRISS' in self.get_text(date):
                date_text = date_text.split("FRISS")[0]
            return DateParser.parse(date_text)

        if not date:
            date = soup.find('span', class_='m-author__catDateTitulusCreateDate')

        if not date:
            date = soup.find('div', class_='m-author__wrapCatDateTitulus')

        return DateParser.parse(self.get_text(date, ''))

    def get_tags(self, soup) -> Set[str]:
        tag = soup.find('a', class_='o-articleHead__catWrap')
        if tag is None:
            # old css class
            tag = soup.find('a', class_='tag')
        return set() if tag is None else {self.get_text(tag)}

    def remove_unnecessary_text_from_article(self, article):
        article = article.replace('Ahol a futball esze és szíve találkozik!', '')
        article = article.replace('Válogatott írások a legjobbaktól.', '')
        article = article.replace('Minden hónapban legalább 24 kiemelkedő színvonalú tartalom havi 990 forintért.', '')
        article = article.replace('Próbáld ki 490 forintért az első hónapban!', '')
        article = article.replace('Ahogyan szerzőgárdánk összetétele ígéri, a legkülönfélébb műfajokban, témákban jelennek majd meg szövegek: lesznek elemzések, interjúk, kisprózák, riportok, lesznek hazai és nemzetközi témák, adat- és véleményalapú cikkek egyaránt.', '')
        article = article.replace('Tartsatok velünk minél többen, hogy együtt alakíthassuk, építhessük az új magyar futballújságot, amely leginkább az írások nívójával és egyediségével akar kitűnni!', '')
        article = article.replace('Fizessetek elő, olvassatok, kövessetek minket, vitatkozzatok velünk, adjatok ötleteket!', '')
        article = article.replace('Várjuk ebbe a közösségbe mindazokat, akiknek a futball csak egy játék, s mindazokat, akiknek a futball több mint egy játék.', '')
        article = article.replace('KIPRÓBÁLOM!', '')
        article = article.replace('Sőt, már a Tumblren is megtalálsz minket!', '')
        article = article.replace('Kérjük, tartsanak minél többen velünk, hogy együtt alakíthassuk, építhessük az új magyar futballújságot, amely leginkább az írások nívójával és egyediségével akar kitűnni. Fizessenek elő, olvassanak, kövessenek minket, vitatkozzanak velünk, adjanak ötleteket.', '')
        article = article.replace('Sőt már a Tumblren is megtalálsz minket!', '')
        article = article.replace('Kövess minket a Facebookon is!', '')
        return article

    def get_html_tags_to_remove(self, soup) -> List[Tag]:
        to_remove = []
        to_remove.extend(soup.find_all('blockquote', class_='instagram-media'))
        to_remove.extend(soup.find_all('blockquote', class_='twitter-tweet'))
        to_remove.extend(soup.find_all('div', class_='fb-post'))
        to_remove.extend(soup.find_all('span', class_='a-imgSource'))
        to_remove.extend(soup.find_all('div', class_='m-authorRecommend'))
        to_remove.extend(soup.find_all('div', class_='o-cegPostCnt'))
        to_remove.extend(soup.find_all('div', class_='article-recommendation-container'))
        to_remove.extend(soup.find_all('div', class_='wpb_content_element'))
        to_remove.extend(soup.find_all('div', class_='m-newsletter'))
        to_remove.extend(soup.find_all('figure', class_='wp-caption'))
        # mischievous pages contains text in the sidebar
        to_remove.extend(soup.find_all('div', class_='sidebar'))
        to_remove.extend(soup.find_all('span', class_='category titulus'))
        to_remove.extend(soup.find_all('div', class_='ad-container'))
        to_remove.extend(soup.find_all('div', class_='central-wp-gallery-wrapper'))
        to_remove.extend(soup.find_all('div', class_='survey'))
        to_remove.extend(soup.find_all('iframe'))

        return to_remove
