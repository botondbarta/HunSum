from summarization.html_parsers.hvg_parser import HvgParser
from summarization.html_parsers.index_parser import IndexParser
from summarization.html_parsers.metropol_parser import MetropolParser
from summarization.html_parsers.nlc_parser import NLCParser
from summarization.html_parsers.parser_24 import Parser24
from summarization.html_parsers.parser_base import ParserBase
from summarization.html_parsers.telex_parser import TelexParser


class HtmlParserFactory:
    parsers = {
        'telex':    TelexParser,
        'index':    IndexParser,
        '24':       Parser24,
        'metropol': MetropolParser,
        'nlc':      NLCParser,
        'hvg':      HvgParser,
    }

    @classmethod
    def get_parser(cls, site) -> ParserBase:
        return cls.parsers[site]()
