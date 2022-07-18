from bs4 import BeautifulSoup

from summarization.errors.missing_article_error import MissingArticleError
from summarization.errors.missing_title_error import MissingTitleError
from summarization.html_parsers.parser_base import ParserBase
from summarization.models.article import Article
from summarization.models.page import Page


class TelexParser(ParserBase):
    @staticmethod
    def get_article(page: Page) -> Article:
        html_soup = BeautifulSoup(page.html, 'html.parser')
        title = html_soup.find('div', class_="title-section__top")
        if title is None:
            raise MissingTitleError(page.url)
        lead = html_soup.find('p', class_="article__lead")
        lead = "" if lead is None else lead.text.strip()
        article = html_soup.find('div', class_="article-html-content")
        tags1 = html_soup.findAll('a', class_="tag--meta")
        tags2 = html_soup.findAll('meta', {"name": "article:tag"})
        tags = set(map(lambda t: t.text.strip(), tags1)).union(set(map(lambda t: t['content'], tags2)))
        if article is None:
            raise MissingArticleError(page.url)
        return Article(title.text.strip(), lead, article.text, page.domain, page.url, page.date, tags)

