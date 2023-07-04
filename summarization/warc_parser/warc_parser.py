import re
from datetime import datetime
from typing import Iterator

import tldextract
import warc

from summarization.data_models.page import Page


class WarcParser:
    def __init__(self, bad_index_file=None):
        if bad_index_file:
            with open(bad_index_file) as f:
                self.bad_index = re.compile('^{}$'.format('|'.join(
                    r'(?:(http|https)://(www\.)?{})'.format(line.strip()) for line in f)))

    def iter_pages(self, file) -> Iterator[Page]:
        warc_file = warc.open(file)
        for record in warc_file:
            url = record['WARC-Target-URI']
            date = datetime.strptime(record['WARC-Date'], '%Y-%m-%dT%H:%M:%SZ')
            html_header, html_text = record.payload.read().split(b'\r\n\r\n', maxsplit=1)
            ext = tldextract.extract(url)
            domain = f'{ext.domain}.{ext.suffix}'
            if not self.bad_index.match(url):
                try:
                    yield Page(url, domain, date, html_text.decode('utf-8'))
                except:
                    yield Page(url, domain, date, html_text.decode('latin-1'))
