from summarization.errors.page_error import PageError


class InvalidPageError(PageError):
    def __init__(self, url, msg=''):
        message = f'Invalid page: {url}\n{msg}'
        super().__init__(message)
