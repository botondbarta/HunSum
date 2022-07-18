class MissingTitleError(Exception):
    def __init__(self, url):
        message = f'Missing title in {url}'
        super().__init__(message)
