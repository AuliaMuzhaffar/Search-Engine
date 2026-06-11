from typing import Set, Tuple, List
import logging
from pathlib import Path
from config.settings import settings

logger = logging.getLogger(__name__)

class IndonesianSpeller:
    def __init__(self, vocabulary: Set[str] = None):
        self.vocabulary = vocabulary if vocabulary is not None else set()
        if not self.vocabulary:
            self.vocabulary = self._load_vocabulary_from_index()

    def _load_vocabulary_from_index(self) -> Set[str]:
        """
        Dynamically extracts all terms from the generated inverted index.
        """
        index_file = settings.INDEX_DIR / "inverted_index.txt"
        vocab = set()
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if ':' in line:
                            term, _ = line.split(':', 1)
                            vocab.add(term.strip())
                logger.info(f"Loaded {len(vocab)} words into speller vocabulary from index.")
            except Exception as e:
                logger.error(f"Error loading vocabulary for speller: {e}")
        return vocab

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculates the Levenshtein distance between two strings.
        """
        if len(s1) < len(s2):
            return IndonesianSpeller.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (0 if c1 == c2 else 1)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def correct_word(self, word: str) -> str:
        """
        Finds the closest word in the vocabulary. Returns original word if no close match.
        """
        word = word.lower()
        if not word or word in self.vocabulary or word.isdigit():
            return word

        closest_word = word
        min_distance = 3  # Max threshold of distance is 2

        for vocab_word in self.vocabulary:
            # Quick optimization: check length differences
            if abs(len(vocab_word) - len(word)) >= min_distance:
                continue
            
            dist = self.levenshtein_distance(word, vocab_word)
            if dist < min_distance:
                min_distance = dist
                closest_word = vocab_word

        return closest_word

    def correct_query(self, query: str) -> Tuple[str, bool]:
        """
        Parses query, corrects spelling of each word, and returns (corrected_query, has_changes).
        """
        if not query:
            return "", False

        words = query.split()
        corrected_words = []
        has_changes = False

        for word in words:
            # Keep punctuation at the end of word if any (e.g. "covid,")
            clean_word = "".join(c for c in word if c.isalnum()).lower()
            if not clean_word:
                corrected_words.append(word)
                continue
                
            corrected = self.correct_word(clean_word)
            
            if corrected != clean_word:
                has_changes = True
                # Reconstruct word preserving original case/punctuation if needed
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)

        return " ".join(corrected_words), has_changes
