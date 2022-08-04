from setuptools import setup

setup(
        name='summarization',
        version='0.1',
        packages=['utils', 'errors', 'models', 'entrypoints', 'serializers', 'warc_parser', 'html_parsers'],
        package_dir={'': 'summarization'},
        url='',
        license='',
        author='Dorina Lakatos, Botond Barta',
        author_email='',
        description='Hungarian summarization'
)
