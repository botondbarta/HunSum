import gzip
from abc import ABC
from typing import List

from summarization.data_models.article import Article


class ArticleSerializer(ABC):
    @staticmethod
    def serialize_articles(file_to_save_to, articles: List[Article]):
        with gzip.open(file_to_save_to, "a") as outfile:
            for article in articles:
                outfile.write(article.to_json().encode())
                outfile.write('\n'.encode())
