from bs4 import BeautifulSoup

from summarization.html_parsers.parser_base import ParserBase
from summarization.utils.article import Article
from summarization.utils.page import Page


class TelexParser(ParserBase):
    @staticmethod
    def get_article(page: Page) -> Article:
        html_soup = BeautifulSoup(page.html, 'html.parser')
        title = html_soup.title
        leads = html_soup.findAll('p', attrs={"class": "article__lead"})
        if len(leads) != 1:
            # TODO
            return None
        articles = html_soup.findAll('div', attrs={"class": "article-html-content"})
        if len(articles) != 1:
            # TODO
            return None
        return Article(title, leads[0], articles[0], page.domain, page.url, page.date)

