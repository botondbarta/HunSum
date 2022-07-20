#!/bin/bash

CC_CORPUS_PATH=$HOME/PycharmProjects/cc_corpus

CC_CORPUS_SCRIPT_PATH=$CC_CORPUS_PATH/scripts
ALLOWED_MIMES_PATH=$CC_CORPUS_PATH/allowed_mimes.txt

while read -r LINE
do
  IFS='|' read -ra INDEX <<< "$LINE"

  SITE="${INDEX[0]}"
  URL="${INDEX[1]}"

  echo "Downloading $SITE from CommonCrawl ($URL)"

  python "$CC_CORPUS_SCRIPT_PATH"/get_indexfiles.py \
    -q $URL \
    -o "$HOME/CommonCrawl/$SITE/cc_index" \
    -l "$SITE.log" \
    -m 5

  echo 'Filtering index'
  python "$CC_CORPUS_SCRIPT_PATH"/filter_index.py \
    "$HOME/CommonCrawl/$SITE/cc_index/" \
    "$HOME/CommonCrawl/$SITE/cc_index_filtered/" \
    -a "$ALLOWED_MIMES_PATH" \
    -P 12

  echo 'Deduplicating index'
  python "$CC_CORPUS_SCRIPT_PATH"/deduplicate_index_urls.py \
    -i "$HOME/CommonCrawl/$SITE/cc_index_filtered/" \
    -o "$HOME/CommonCrawl/$SITE/cc_index_dedup/"

  echo 'Downloading pages'
  python "$CC_CORPUS_SCRIPT_PATH"/download_pages.py \
    -o "$HOME/CommonCrawl/$SITE/cc_downloaded" \
    -e warc.gz \
    -i "$HOME/CommonCrawl/$SITE/cc_index_dedup/*.gz"

done < "$1"