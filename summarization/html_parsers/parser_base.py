from abc import ABC, abstractmethod

from summarization.models.page import Page
from summarization.models.article import Article


class ParserBase(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def get_article(page: Page) -> Article:
        raise NotImplementedError
