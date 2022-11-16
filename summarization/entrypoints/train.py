import click

from summarization.models.bert2bert import Bert2Bert
from summarization.models.mt5 import MT5


@click.command()
@click.argument('model_type')
@click.argument('config_path')
@click.option('--evaluate_only', is_flag=True, default=False)
def main(model_type, config_path, evaluate_only):
    if model_type == 'mt5':
        model = MT5(config_path)
    else:
        model = Bert2Bert(config_path)
    model.full_train(do_train=not evaluate_only, generate_predict=not evaluate_only)


if __name__ == '__main__':
    main()
