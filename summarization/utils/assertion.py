from summarization.errors.missing_article_error import MissingArticleError
from summarization.errors.missing_title_error import MissingTitleError


def assert_has_article(article, url):
    if article is None:
        raise MissingArticleError(url)


def assert_has_title(title, url):
    if title is None:
        raise MissingTitleError(url)
