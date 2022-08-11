from summarization.errors.page_error import PageError


class InvalidPageError(PageError):
    def __init__(self, url, msg=''):
        message = f'Invalid page ({msg}): {url}'
        super().__init__(message)
