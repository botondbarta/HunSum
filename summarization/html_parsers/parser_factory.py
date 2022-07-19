from summarization.html_parsers.parser_base import ParserBase
from summarization.html_parsers.telex_parser import TelexParser


class HtmlParserFactory:
    parsers = {
        'telex': TelexParser,
    }

    @classmethod
    def get_parser(cls, site) -> ParserBase:
        return cls.parsers[site]()
