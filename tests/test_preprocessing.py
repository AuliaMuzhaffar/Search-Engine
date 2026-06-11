import pytest
from pathlib import Path
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.tokenizer import Tokenizer

def test_cleaner_removes_punctuation_and_symbols():
    cleaner = TextCleaner()
    # Mocking stopwords to keep tests simple
    cleaner.stopwords = {"dan", "yang"}
    
    text = "Halo! Ini adalah contoh teks, dan & (123) untuk dibersihkan."
    cleaned = cleaner.clean(text)
    
    # Check that punctuation, numbers and symbols are removed
    assert "!" not in cleaned
    assert "&" not in cleaned
    assert "123" not in cleaned
    assert "," not in cleaned
    # Stopwords like "dan" should be removed
    assert "dan" not in cleaned

def test_tokenizer_splits_words():
    tokenizer = Tokenizer()
    text = "saya belajar machine learning"
    tokens = tokenizer.tokenize(text)
    assert tokens == ["saya", "belajar", "machine", "learning"]

def test_stemmer_indonesian():
    tokenizer = Tokenizer()
    tokens = ["memakan", "minuman", "berjalan"]
    stemmed = tokenizer.stem(tokens)
    assert stemmed == ["makan", "minum", "jalan"]
