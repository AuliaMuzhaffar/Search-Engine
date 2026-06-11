import pytest
from pathlib import Path
from src.indexing.inverted_index import InvertedIndex

def test_inverted_index_build_and_persistence(tmp_path):
    # Setup a temp index directory
    index_builder = InvertedIndex(index_dir=tmp_path)
    
    # Create a dummy processed doc
    doc_dir = tmp_path / "processed"
    doc_dir.mkdir()
    doc_file = doc_dir / "doc1.txt"
    doc_file.write_text("makan nasi goreng nasi")
    
    index_builder.build_global_index(doc_dir)
    
    assert "nasi" in index_builder.index
    assert "goreng" in index_builder.index
    
    # Check positions
    # tokens are: ["makan", "nasi", "goreng", "nasi"]
    # indices: 0: makan, 1: nasi, 2: goreng, 3: nasi
    assert index_builder.index["nasi"] == [(0, 1), (0, 3)]
    assert index_builder.index["goreng"] == [(0, 2)]
    assert index_builder.doc_names[0] == "doc1.txt"
    
    # Save the index
    index_builder.save()
    
    # Load into a new index instance
    new_index = InvertedIndex(index_dir=tmp_path)
    success = new_index.load()
    
    assert success
    assert "nasi" in new_index.index
    assert new_index.index["nasi"] == [(0, 1), (0, 3)]
    assert new_index.doc_names[0] == "doc1.txt"
