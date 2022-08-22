from panflute import Image, toJSONFilters


def filter_images(elem, doc):
    if type(elem) == Image:
        return []


if __name__ == "__main__":
    toJSONFilters([filter_images])
