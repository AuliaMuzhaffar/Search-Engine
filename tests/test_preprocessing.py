import pytest
from pathlib import Path
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.tokenizer import Tokenizer
from src.preprocessing.speller import IndonesianSpeller

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

def test_levenshtein_distance():
    assert IndonesianSpeller.levenshtein_distance("kucing", "kucing") == 0
    assert IndonesianSpeller.levenshtein_distance("kucing", "kuceng") == 1
    assert IndonesianSpeller.levenshtein_distance("ekonomy", "ekonomi") == 1
    assert IndonesianSpeller.levenshtein_distance("ekonomy", "ekonom") == 1

def test_speller_corrections():
    vocab = {"vaksin", "covid", "ekonomi", "indonesia", "bpjs"}
    speller = IndonesianSpeller(vocabulary=vocab)
    
    corrected, has_changes = speller.correct_query("vaksen cofed")
    assert has_changes
    assert corrected == "vaksin covid"
    
    # Unchanged
    corrected, has_changes = speller.correct_query("indonesia ekonomi")
    assert not has_changes
    assert corrected == "indonesia ekonomi"

