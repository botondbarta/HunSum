from quntoken import tokenize


class Tokenizer:
    @staticmethod
    def sentence_tokenize(text: str):
        doc = tokenize(text, mode='sentence')
        return [sent.replace('\n', '') for sent in doc if sent != '\n']

    @staticmethod
    def count_sentences(text: str) -> int:
        doc = tokenize(text, mode='sentence')
        return len([sentence for sentence in doc if sentence != '\n'])

    @staticmethod
    def count_tokens(text: str) -> int:
        doc = tokenize(text)
        return len([token for token in doc if token != '\n'])
