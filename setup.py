from setuptools import setup

setup(
    name='summarization',
    version='0.1',
    description='',
    url='',
    packages=['summarization'],
    license='',
    author='Dorina Lakatos, Botond Barta',
    python_requires='',
    install_requires=[
        'setuptools~=61.2.0',
        'dateparser~=1.1.1',
        'typing~=3.7.4.3',
        'click~=8.0.4',
        'tqdm~=4.64.0',
        'tldextract~=3.3.1',
        'beautifulsoup4~=4.11.1',
        'pypandoc~=1.8.1',
        'warc3-wet',
    ]
)
