import click

from summarization.preprocess.document_embedder import DocumentEmbedder


@click.command()
@click.argument('config_path')
@click.option('--sites', default='all', help='Sites to calculate lead-article similarities for, separated by commas')
def main(config_path, sites):
    doc_embedder = DocumentEmbedder(config_path)
    doc_embedder.calculate_doc_similarity_for_sites(sites)


if __name__ == '__main__':
    main()
