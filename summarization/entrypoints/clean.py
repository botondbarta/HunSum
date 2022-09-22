import click

from summarization.preprocess.article_cleaner import ArticleCleaner


@click.command()
@click.argument('config_path')
def main(config_path):
    cleaner = ArticleCleaner(config_path)
    cleaner.clean_articles()


if __name__ == '__main__':
    main()
