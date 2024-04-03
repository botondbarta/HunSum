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

## Citation
If you use our dataset or models, please cite the following paper:

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
