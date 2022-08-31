import click

from summarization.preprocess.preprocessor import Preprocessor


@click.command()
@click.argument('config_path')
def main(config_path):
    preprocessor = Preprocessor(config_path)
    preprocessor.preprocess()


if __name__ == '__main__':
    main()
