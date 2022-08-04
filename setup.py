from setuptools import find_packages, setup

setup(
        name='summarization',
        version='',
        packages=find_packages(exclude=['scripts']),
        package_dir={'': 'summarization'},
        url='',
        license='',
        author='Dorina Lakatos, Botond Barta',
        author_email='',
        description='',
        install_requires=[
            'tqdm',
            'click',
            'dateparser',
            'typing',
            'warc3-wet',
            'tldextract',
            'beautifulsoup4',
        ],
)
