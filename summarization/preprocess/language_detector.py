import fasttext


class LanguageDetector:
    def __init__(self, model_path='/tmp/lid.176.bin'):
        self.model = fasttext.load_model(model_path)

    def predict(self, text):
        # Extract ISO language code from model response
        return self.model.predict(text)[0][0].rpartition('__')[-1]
