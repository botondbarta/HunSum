from os import listdir, mkdir, path

import click
from tqdm import tqdm

from errors.missing_article_error import MissingArticleError
from errors.missing_lead_error import MissingLeadError
from errors.missing_title_error import MissingTitleError

from summarization.errors.page_error import PageError
from summarization.html_parsers.parser_factory import HtmlParserFactory
from summarization.serializers.article_serializer import ArticleSerializer
from summarization.utils.logger import get_logger
from summarization.warc_parser.warc_parser import WarcParser


@click.command()
@click.argument('src_directory')
@click.argument('out_directory')
@click.option('--sites', default='all', help='Sites to scrape, separated by commas')
def main(src_directory, out_directory, sites):
    warc_parser = WarcParser('bad_index.txt')

    make_out_dir_if_not_exists(out_directory)

    sites_to_scrape = listdir(src_directory) if sites == 'all' else sites.split(',')
    for site in sites_to_scrape:
        articles = []
        parser = HtmlParserFactory.get_parser(site)

        log_file = f'{path.join(out_directory, site)}.log.txt'
        logger = get_logger(site, log_file)

        logger.info(f'Started processing pages for: {site} with {type(parser).__name__}')

        subdirectory = path.join(src_directory, f'{site}/cc_downloaded')
        for file_name in listdir(subdirectory):
            logger.info(f'Parsing file: {file_name}')
            file_path = path.join(subdirectory, file_name)
            for page in tqdm(warc_parser.iter_pages(file_path)):
                try:
                    article = parser.get_article(page)
                    articles.append(article)
                except PageError as e:
                    logger.warning(e)
                except Exception as e:
                    logger.error(e, f'in {page.url}')

        ArticleSerializer.serialize_articles(out_directory, site, articles)


def make_out_dir_if_not_exists(out_directory):
    if not path.exists(out_directory):
        mkdir(out_directory)


if __name__ == '__main__':
    main()
