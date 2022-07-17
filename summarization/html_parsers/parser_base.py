from abc import ABC, abstractmethod

from summarization.utils.page import Page
from utils.article import Article


class ParserBase(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def get_article(page: Page) -> Article:
        raise NotImplementedError
