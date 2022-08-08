from summarization.errors.page_error import PageError


class MissingLeadError(PageError):
    def __init__(self, url):
        message = f'Missing lead in {url}'
        super().__init__(message)
