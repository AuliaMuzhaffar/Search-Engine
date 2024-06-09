import os
from rank_bm25 import BM25Okapi
import json

# Mendefinisikan path ke direktori dokumen
documents_folder = r'D:/kuliah/Semester 5/Penelusuran Informasi/PI Project 2.0/output_folder'

# Inisialisasi list untuk menyimpan teks dari dokumen
documents = []

# Loop melalui semua file dalam direktori dan baca teksnya
for file_name in os.listdir(documents_folder):
    if file_name.endswith('.txt'):
        file_path = os.path.join(documents_folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            document_text = file.read()
            documents.append(document_text)

# Membuat korpus
tokenized_documents = [document.split() for document in documents]
bm25 = BM25Okapi(tokenized_documents)

# Membaca file urldoc.json
with open('urldoc.json', 'r', encoding='utf-8') as json_file:
    urldoc = json.load(json_file)

# User memasukkan query
query = input("Masukkan query: ")

# Tokenisasi query
query_tokens = query.split()

# Menghitung skor BM25
bm25_scores = bm25.get_scores(query_tokens)

# Mendapatkan indeks dokumen dengan skor tertinggi
top_document_indices = bm25_scores.argsort()[::-1]

# Menampilkan hasil
print(f"Hasil pencarian dari BM25:")
num_matched_documents = 0

for rank, idx in enumerate(top_document_indices, start=1):
    bm25_score = bm25_scores[idx]
    if bm25_score > 0:  # Memeriksa apakah similarity lebih besar dari 0
        document_title = os.listdir(documents_folder)[idx]
        doc_id = document_title.split('_')[0]  # Dapatkan dokumen ID dari nama file
        print(f"Rank Dokumen: {rank}")
        print(f"Judul Dokumen: {document_title}")
        print(f"Similarity: {bm25_score}")
        print(f"link url: {urldoc[document_title]}")  # Cetak URL sesuai dengan nama file
        num_matched_documents += 1
        print()

# Menampilkan jumlah dokumen yang cocok dengan query
print(f"Jumlah dokumen yang cocok dengan query: {num_matched_documents}")
