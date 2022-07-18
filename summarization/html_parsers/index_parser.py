from html_parsers.parser_base import ParserBase
from models.article import Article
from models.page import Page


class IndexParser(ParserBase):
    @staticmethod
    def get_article(page: Page) -> Article:
        raise NotImplementedError
