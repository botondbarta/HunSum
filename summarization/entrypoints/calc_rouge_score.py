import click
import datasets
import evaluate


@click.command()
@click.argument('references')  # path to original
@click.argument('predictions')  # path to predicted
def main(references, predictions):
    rouge = datasets.load_metric("rouge")
    # rouge = evaluate.load("rouge")
    with open(references, 'r') as ref_file, open(predictions, 'r') as pred_file:
        refs = ref_file.readlines()
        preds = pred_file.readlines()
        rouge_output = rouge.compute(
            predictions=preds, references=refs, rouge_types=["rouge2"]
        )["rouge2"].mid
        print(rouge_output)


if __name__ == '__main__':
    main()
