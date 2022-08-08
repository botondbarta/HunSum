from summarization.errors.page_error import PageError


class MissingTitleError(PageError):
    def __init__(self, url):
        message = f'Missing title in {url}'
        super().__init__(message)
