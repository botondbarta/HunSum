import click

from summarization.preprocess.document_embedder import DocumentEmbedder


@click.command()
@click.argument('input_dir')
@click.argument('output_dir')
@click.option('--sites', default='all', help='Sites to clean, separated by commas')
def main(input_dir, output_dir, sites):
    doc_embedder = DocumentEmbedder(input_dir, output_dir)
    doc_embedder.create_doc_embeddings_for_sites(sites)


if __name__ == '__main__':
    main()
