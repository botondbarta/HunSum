class InvalidPageError(Exception):
    def __init__(self, url):
        message = f'Invalid page: {url}'
        super().__init__(message)
