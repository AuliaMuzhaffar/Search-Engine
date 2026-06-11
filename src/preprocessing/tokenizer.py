from typing import List
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class Tokenizer:
    def __init__(self):
        self.stemmer_factory = StemmerFactory()
        self.stemmer = self.stemmer_factory.create_stemmer()

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return text.split()

    def stem(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(token) for token in tokens if token]

    def tokenize_and_stem(self, text: str) -> List[str]:
        tokens = self.tokenize(text)
        return self.stem(tokens)
