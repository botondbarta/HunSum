from abc import ABC, abstractmethod

from summarization.utils.page import Page


class ParserBase(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def get_article(page: Page):
        raise NotImplementedError
