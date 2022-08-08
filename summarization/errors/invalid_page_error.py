class InvalidPageError(Exception):
    def __init__(self, url, msg=''):
        message = f'Invalid page: {url}\n{msg}'
        super().__init__(message)
