import click

from summarization.preprocess.article_cleaner import ArticleCleaner


@click.command()
@click.argument('config_path')
@click.option('--sites', default='all', help='Sites to clean, separated by commas')
def main(config_path, sites):
    cleaner = ArticleCleaner(config_path)
    cleaner.clean_articles(sites)


if __name__ == '__main__':
    main()
