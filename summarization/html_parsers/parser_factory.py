from html_parsers.hvg_parser import HVGParser
from html_parsers.index_parser import IndexParser
from html_parsers.napi_parser import NapiParser
from html_parsers.nepszava_parser import NepszavaParser
from html_parsers.parser_base import ParserBase
from html_parsers.telex_parser import TelexParser
from html_parsers.twentyfour_parser import TwentyFourParser


class HtmlParserFactory:
    parsers = {
        'telex': lambda: TelexParser(),
        'index': lambda: IndexParser(),
        'napi': lambda: NapiParser(),
        'hvg': lambda: HVGParser(),
        '24': lambda: TwentyFourParser(),
        'nepszava': lambda: NepszavaParser(),
    }

    @classmethod
    def get_parser(cls, site) -> ParserBase:
        return cls.parsers[site]()
