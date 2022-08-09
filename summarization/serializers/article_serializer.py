import gzip
import os.path
from abc import ABC
from typing import List

from summarization.models.article import Article


class ArticleSerializer(ABC):
    @staticmethod
    def serialize_articles(directory: str, news_page_name: str, articles: List[Article]):
        file = os.path.join(directory, f'{news_page_name}.jsonl.gz')
        with gzip.open(file, "a") as outfile:
            for article in articles:
                outfile.write(article.to_json().encode())
                outfile.write('\n'.encode())
