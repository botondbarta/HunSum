# summarization

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

## Usage
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
python entrypoints/run_parse_warc_pages.py ../../CommonCrawl ../../articles preprocess.yaml
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