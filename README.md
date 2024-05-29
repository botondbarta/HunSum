# Abstractive and extractive summarization for Hungarian
Links to the HunSum-2 dataset and our baseline models:
- [HunSum-2-abstractive](https://huggingface.co/datasets/SZTAKI-HLT/HunSum-2-abstractive)
- [HunSum-2-extractive](https://huggingface.co/datasets/SZTAKI-HLT/HunSum-2-extractive)
- [mt5-base](https://huggingface.co/SZTAKI-HLT/mT5-base-HunSum-2)
- [Bert2Bert](https://huggingface.co/SZTAKI-HLT/Bert2Bert-HunSum-2)

Links to the HunSum-1 dataset and our baseline models:
- [HunSum-1](https://huggingface.co/datasets/SZTAKI-HLT/HunSum-1)
- [mt5-base](https://huggingface.co/SZTAKI-HLT/mT5-base-HunSum-1)
- [mt5-small](https://huggingface.co/SZTAKI-HLT/mT5-small-HunSum-1)
- [Bert2Bert](https://huggingface.co/SZTAKI-HLT/Bert2Bert-HunSum-1)

## Setup
```bash
conda create --name my-env python=3.8.13
conda activate my-env

conda install -c conda-forge pandoc
pip install -e .
```

#### Install LSH package used for deduplication
```bash
git clone https://github.com/mattilyra/LSH
cd LSH
git checkout fix/filter_duplicates
pip install -e .
```

## How to create the corpus
### Download data from Common Crawl
#### Install CommonCrawl Downloader
```bash
git clone git@github.com:DavidNemeskey/cc_corpus.git
cd cc_corpus
pip install -e .
```
#### Download data
Arguments:
* text file containing the urls to download: `indexes_to_download.txt`
* path of the cc_corpus
* output directory
```bash
scripts/download_data.sh indexes_to_download.txt ../cc_corpus/ ../CommonCrawl/
```
### Parse articles
Arguments:
* downloaded data
* output directory
* config file

The cleaned articles will be in the config.clean_out_dir 
```bash
cd summarization
python entrypoints/parse_warc_pages.py ../../CommonCrawl ../../articles preprocess.yaml
```

### Calculate document embeddings for leads and articles for cleaning
Arguments:
* config file
```bash
cd summarization
python entrypoints/calc_doc_similarities.py preprocess.yaml
```

### Clean articles
Arguments:
* config file
```bash
cd summarization
python entrypoints/clean.py preprocess.yaml
```

### Deduplicate articles
Arguments:
* config file
```bash
cd summarization
python entrypoints/deduplicate.py preprocess.yaml
```

## How to add your own parser
To add a new parser for the corpus creation process, follow these steps:
#### 1. Create a new parser class for the specific website.
Place the parser in the [html_parsers](https://github.com/botondbarta/HunSum/tree/main/summarization/html_parsers) package. The parser should inherit from the ParserBase class and implement the following methods:
```python
class MyNewWebsiteParser(ParserBase):
    def check_page_is_valid(self, url, soup):
        # Implement logic to check if the page is valid
        # (e.g. check if the page is a gallery page if it's not indicated by the URL)
        # if needed raise InvalidPageError(url, 'problem description')

    def get_title(self, url, soup) -> str:
        # Implement logic to extract title

    def get_lead(self, soup) -> str:
        # Implement logic to extract lead

    def get_article_text(self, url, soup) -> str:
        # Implement logic to extract the main article text

    def get_date_of_creation(self, soup) -> Optional[datetime]:
        # Implement logic to extract the date of creation

    def get_tags(self, soup) -> Set[str]:
        # Implement logic to extract tags

    def get_html_tags_to_remove(self, soup) -> List[Tag]:
        # Implement logic to specify which HTML tags to remove

    def remove_unnecessary_text_from_article(self, article) -> str:
        # Implement logic to remove unnecessary text from the article (e.g. ads that cannot be removed by HTML tags)
        return article
```
#### 2. Register your parser in the [HtmlParserFactory](https://github.com/botondbarta/HunSum/blob/main/summarization/html_parsers/parser_factory.py#L16).
```python
class HtmlParserFactory:
    parsers = {
        ...
        'mywebsite': MyNewWebsiteParser  # Register your new parser here
        ...
    }
```
You're all set to start parsing your articles with the [parse_warc_pages.py](https://github.com/botondbarta/HunSum/blob/main/summarization/entrypoints/parse_warc_pages.py) script. If you only want to parse your new website, just use the `--sites mywebsite` option.

## Citation
If you use our dataset or models, please cite the following papers:

```
@inproceedings {HunSum-1,
    title = {{HunSum-1: an Abstractive Summarization Dataset for Hungarian}},
    booktitle = {XIX. Magyar Számítógépes Nyelvészeti Konferencia (MSZNY 2023)},
    year = {2023},
    publisher = {Szegedi Tudományegyetem, Informatikai Intézet},
    address = {Szeged, Magyarország},
    author = {Barta, Botond and Lakatos, Dorina and Nagy, Attila and Nyist, Mil{\'{a}}n Konor and {\'{A}}cs, Judit},
    pages = {231--243}
}
```
```
@inproceedings{barta-etal-2024-news-summaries,
    title = "From News to Summaries: Building a {H}ungarian Corpus for Extractive and Abstractive Summarization",
    author = "Barta, Botond  and
      Lakatos, Dorina  and
      Nagy, Attila  and
      Nyist, Mil{\'a}n Konor  and
      {\'A}cs, Judit",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.662",
    pages = "7503--7509",
    abstract = "Training summarization models requires substantial amounts of training data. However for less resourceful languages like Hungarian, openly available models and datasets are notably scarce. To address this gap our paper introduces an open-source Hungarian corpus suitable for training abstractive and extractive summarization models. The dataset is assembled from segments of the Common Crawl corpus undergoing thorough cleaning, preprocessing and deduplication. In addition to abstractive summarization we generate sentence-level labels for extractive summarization using sentence similarity. We train baseline models for both extractive and abstractive summarization using the collected dataset. To demonstrate the effectiveness of the trained models, we perform both quantitative and qualitative evaluation. Our models and dataset will be made publicly available, encouraging replication, further research, and real-world applications across various domains.",
}
```
