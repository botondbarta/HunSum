import logging
from os import listdir, mkdir, path

import click
from tqdm import tqdm

from summarization.html_parsers.parser_factory import HtmlParserFactory
from summarization.serializers.article_serializer import ArticleSerializer
from summarization.utils.logger import get_logger
from summarization.warc_parser.warc_parser import WarcParser

logger = get_logger(__name__)


@click.command()
@click.argument('src_directory')
@click.argument('out_directory')
def main(src_directory, out_directory):
    warc_parser = WarcParser('bad_index.txt')

    if not path.exists(out_directory):
        mkdir(out_directory)

    for news_page in listdir(src_directory):
        articles = []
        parser = HtmlParserFactory.get_parser(news_page)

        logging.basicConfig(filename=f'{path.join(out_directory, news_page)}.log.txt', level=logging.INFO,
                            format='%(asctime)s %(levelname)-19s %(message)s', filemode="w")
        logger.info(f'Started processing pages for: {news_page} with {type(parser).__name__}')

        subdirectory = path.join(src_directory, f'{news_page}/cc_downloaded')
        for file_name in listdir(subdirectory):
            logger.info(f'Parsing file: {file_name}')
            file_path = path.join(subdirectory, file_name)
            for page in tqdm(warc_parser.iter_pages(file_path)):
                try:
                    article = parser.get_article(page)
                    articles.append(article)
                except Exception as e:
                    logger.warning(e)

        ArticleSerializer.serialize_articles(out_directory, news_page, articles)


if __name__ == '__main__':
    main()
