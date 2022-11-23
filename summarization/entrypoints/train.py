import click

from summarization.models.bert2bert import Bert2Bert
from summarization.models.mt5 import MT5


@click.command()
@click.argument('model_type')
@click.argument('config_path')
@click.option('--evaluate_only', is_flag=True, default=False)
@click.option('--evaluate_with_generate', is_flag=True, default=False)
def main(model_type, config_path, evaluate_only, evaluate_with_generate):
    if evaluate_only and evaluate_with_generate:
        print('evaluate_only and evaluate_with_generate cannot be set at once!')
        return
    if model_type == 'mt5':
        model = MT5(config_path)
    else:
        model = Bert2Bert(config_path)
    model.full_train(do_train=not evaluate_only and not evaluate_with_generate,
                     generate_predict=(not evaluate_only) or evaluate_with_generate)


if __name__ == '__main__':
    main()
