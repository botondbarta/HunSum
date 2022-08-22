from panflute import Image, toJSONFilters


def increase_header_level(elem, doc):
    if type(elem) == Image:
        return []


if __name__ == "__main__":
    toJSONFilters([increase_header_level])
