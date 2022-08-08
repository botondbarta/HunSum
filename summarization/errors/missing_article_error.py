from summarization.errors.page_error import PageError


class MissingArticleError(PageError):
    def __init__(self, url):
        message = f'Missing article in {url}'
        super().__init__(message)
