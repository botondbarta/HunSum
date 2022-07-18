class MissingLeadError(Exception):
    def __init__(self, url):
        message = f'Missing lead in {url}'
        super().__init__(message)
