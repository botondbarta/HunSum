import click
from rouge import FilesRouge


@click.argument('references')  # path to original
@click.argument('predictions')  # path to predicted
def main(references, predictions):
    files_rouge = FilesRouge()
    scores = files_rouge.get_scores(predictions, references, avg=True)
    print(scores)


if __name__ == '__main__':
    main()
