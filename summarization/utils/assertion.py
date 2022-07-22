from summarization.errors.missing_article_error import MissingArticleError
from summarization.errors.missing_title_error import MissingTitleError


def assert_has_article(article, url):
    if not article:
        raise MissingArticleError(url)


def assert_has_title(title, url):
    if not title:
        raise MissingTitleError(url)
