from os import listdir, path

from tqdm import tqdm
import click

from summarization.html_parsers.parser_factory import HtmlParserFactory
from summarization.serializers.article_serializer import ArticleSerializer
from summarization.utils.logger import get_logger
from summarization.warc_parser.warc_parser import WarcParser

logger = get_logger(__name__)


@click.command()
@click.argument('src_directory')
@click.argument('out_directory')
def main(src_directory, out_directory):
    warc_parser = WarcParser()

    for news_page in listdir(src_directory):
        logger.info(f'Started processing news page: {news_page}')
        articles = []
        parser = HtmlParserFactory.get_parser(news_page)

        subdirectory = path.join(src_directory, news_page)
        for file_name in tqdm(listdir(subdirectory)):
            file_path = path.join(subdirectory, file_name)
            for page in warc_parser.iter_pages(file_path):
                try:
                    article = parser.get_article(page)
                    articles.append(article)
                except Exception as e:
                    logger.warning(e)

        ArticleSerializer.serialize_articles(out_directory, news_page, articles)


if __name__ == '__main__':
    main()
