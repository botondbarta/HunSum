class MissingArticleError(Exception):
    def __init__(self, url):
        message = f'Missing article in {url}'
        super().__init__(message)
