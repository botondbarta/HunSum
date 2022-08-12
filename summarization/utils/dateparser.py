import dateparser


class DateParser:
    @staticmethod
    def parse(date: str):
        return dateparser.parse(date, settings={'TIMEZONE': 'CET'})
