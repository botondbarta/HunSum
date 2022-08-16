import glob
from os import listdir, mkdir, path

import click
from tqdm import tqdm

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
        log_file = get_next_log_file(out_directory, site)
        already_parsed_segments = get_previously_parsed_segments(out_directory, site)
        logger = get_logger(site, log_file)

        try:
            parser = HtmlParserFactory.get_parser(site)
        except:
            logger.warning(f'No parser found for {site}')
            continue

        logger.info(f'Started processing pages for: {site} with {type(parser).__name__}')

        subdirectory = path.join(src_directory, f'{site}/cc_downloaded')
        files_to_scrape = set(listdir(subdirectory)) - set(already_parsed_segments)
        for file_name in files_to_scrape:
            articles = []
            logger.info(f'Parsing file: {file_name}')
            file_path = path.join(subdirectory, file_name)
            for page in tqdm(warc_parser.iter_pages(file_path)):
                try:
                    article = parser.get_article(page)
                    articles.append(article)
                except PageError as e:
                    logger.warning(e)
                except Exception as e:
                    logger.exception(e, f'in {page.url}')
            ArticleSerializer.serialize_articles(out_directory, site, articles)
            logger.info(f'Parsed file: {file_name}')


def get_next_log_file(out_directory, site):
    log_file_base = path.join(out_directory, site)
    if not path.exists(out_directory):
        next_index = 0
    else:
        prev_log_files = [log_file.rpartition('/')[2] for log_file in glob.glob(f'{log_file_base}.log.*')]
        if not prev_log_files:
            next_index = 0
        else:
            indexes = [int(log_file.partition('log.')[2].partition('.txt')[0]) for log_file in prev_log_files]
            indexes.sort()
            next_index = indexes[-1] + 1
    return f'{log_file_base}.log.{next_index}.txt'


def get_previously_parsed_segments(out_directory, site):
    all_parsed_segments = []
    log_file_base = path.join(out_directory, site)
    prev_log_files = glob.glob(f'{log_file_base}.log.*')
    for log_file in prev_log_files:
        with open(log_file, 'r') as file:
            lines = file.read().split('\n')
        parsed_segments = [line.partition('Parsed file: ')[2] for line in lines if 'Parsed file: ' in line]
        all_parsed_segments.extend(parsed_segments)
    return all_parsed_segments


def make_out_dir_if_not_exists(out_directory):
    if not path.exists(out_directory):
        mkdir(out_directory)


if __name__ == '__main__':
    main()
