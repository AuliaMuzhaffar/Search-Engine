import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class InvertedIndex:
    def __init__(self, index_dir: Path = None):
        self.index_dir = index_dir or settings.INDEX_DIR
        self.index: Dict[str, List[Tuple[int, int]]] = {}
        self.doc_names: Dict[int, str] = {}

    def clean_and_tokenize(self, text: str) -> List[str]:
        # Basic formatting to clean and split the stemmed texts in files
        text = text.replace('\n', ' ')
        text = ' '.join(text.split())
        return text.split()

    def build_for_document(self, file_path: Path, doc_id: int) -> Dict[str, List[Tuple[int, int]]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            tokens = self.clean_and_tokenize(text)
            
            doc_index = {}
            for i, token in enumerate(tokens):
                if token not in doc_index:
                    doc_index[token] = []
                doc_index[token].append((doc_id, i))
            return doc_index
        except Exception as e:
            logger.error(f"Error building index for file {file_path}: {e}")
            return {}

    def build_global_index(self, processed_dir: Path):
        """
        Builds the inverted index from all files in the processed_dir directory.
        """
        if not processed_dir.exists():
            logger.warning(f"Processed directory {processed_dir} does not exist. Cannot build index.")
            return

        self.index.clear()
        self.doc_names.clear()
        
        # Get sorted list of files for consistent doc_ids
        files = sorted([f for f in processed_dir.iterdir() if f.is_file() and f.suffix == '.txt'])
        
        for doc_id, file_path in enumerate(files):
            self.doc_names[doc_id] = file_path.name
            doc_index = self.build_for_document(file_path, doc_id)
            
            for term, positions in doc_index.items():
                if term not in self.index:
                    self.index[term] = []
                self.index[term].extend(positions)
                
        logger.info(f"Built global index with {len(self.index)} terms across {len(self.doc_names)} documents.")

    def save(self, filepath: Path = None):
        """
        Saves the inverted index to a file.
        """
        filepath = filepath or self.index_dir / "inverted_index.txt"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for term, positions in sorted(self.index.items()):
                    positions_str = ', '.join(f"({doc_id}, {pos})" for doc_id, pos in positions)
                    f.write(f"{term}: {positions_str}\n")
            
            # Save document names mapping
            doc_names_path = filepath.parent / "doc_names.txt"
            with open(doc_names_path, 'w', encoding='utf-8') as f:
                for doc_id, name in sorted(self.doc_names.items()):
                    f.write(f"{doc_id}: {name}\n")
                    
            logger.info(f"Saved inverted index to {filepath}")
        except Exception as e:
            logger.error(f"Error saving inverted index: {e}")

    def load(self, filepath: Path = None):
        """
        Loads the inverted index and doc names map from files.
        """
        filepath = filepath or self.index_dir / "inverted_index.txt"
        doc_names_path = filepath.parent / "doc_names.txt"
        
        if not filepath.exists() or not doc_names_path.exists():
            logger.warning("Index files do not exist. Cannot load.")
            return False

        self.index.clear()
        self.doc_names.clear()

        try:
            # Load doc names
            with open(doc_names_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        doc_id_str, name = line.strip().split(':', 1)
                        self.doc_names[int(doc_id_str)] = name.strip()

            # Load index
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        term, positions_str = line.strip().split(':', 1)
                        term = term.strip()
                        # Parse coordinates: (0, 1), (0, 5)
                        coords = re.findall(r'\((\d+),\s*(\d+)\)', positions_str)
                        self.index[term] = [(int(doc_id), int(pos)) for doc_id, pos in coords]
            
            logger.info(f"Loaded index with {len(self.index)} terms and {len(self.doc_names)} doc mappings.")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
import re
