import huspacy


class Tokenizer:
    def __init__(self):
        try:
            self.tokenizer = huspacy.load('hu_core_news_trf')
        except OSError as e:
            huspacy.download('hu_core_news_trf')
            self.tokenizer = huspacy.load('hu_core_news_trf')

    def count_sentences(self, text: str) -> int:
        doc = self.tokenizer(text)
        return len(list(doc.sents))

    def count_tokens(self, text: str) -> int:
        doc = self.tokenizer(text)
        return len([token for token in doc if not token.is_punct])
