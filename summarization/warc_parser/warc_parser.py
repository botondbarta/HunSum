from typing import Iterator
import warc
import tldextract
from datetime import datetime

from summarization.models.page import Page


class WarcParser:
    def __init__(self):
        pass

    def iter_pages(self, file) -> Iterator[Page]:
        warc_file = warc.open(file)
        for record in warc_file:
            url = record['WARC-Target-URI']
            date = datetime.strptime(record['WARC-Date'], '%Y-%m-%dT%H:%M:%SZ')
            html_header, html_text = record.payload.read().split(b'\r\n\r\n', maxsplit=1)
            ext = tldextract.extract(url)
            domain = f'{ext.domain}.{ext.suffix}'
            yield Page(url, domain, date, html_text)

