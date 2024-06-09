import os
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Definisikan path ke direktori dokumen
documents_folder = r'D:/kuliah/Semester 5/Penelusuran Informasi/PI Project 2.0/output_folder'

# Step 2: Inisialisasi list untuk menyimpan teks dari dokumen
documents = []

# Loop melalui semua file dalam direktori dan baca teksnya
for file_name in os.listdir(documents_folder):
    if file_name.endswith('.txt'):
        file_path = os.path.join(documents_folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            document_text = file.read()
            documents.append(document_text)

# Step 3: Inisialisasi objek BM25
tokenized_documents = [document.split() for document in documents]
bm25 = BM25Okapi(tokenized_documents)

# User memasukkan query
query = input("Masukkan query: ")

# Step 4: Inisialisasi objek TF-IDF dan mengubah query menjadi vektor TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
query_vector = tfidf_vectorizer.transform([query])

# Step 5: Menghitung skor TF-IDF
tfidf_scores = (query_vector * tfidf_matrix.T).A[0]

# Step 6: Menghitung skor BM25
query_tokens = query.split()
bm25_scores = bm25.get_scores(query_tokens)

# Step 7: Menggabungkan skor dari kedua algoritma
combined_scores = [0.7*bm25_scores[i] + 0.3*tfidf_scores[i] for i in range(len(bm25_scores))]

# Step 8: Mendapatkan indeks dokumen dengan skor tertinggi
top_document_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)

# Step 9: Menampilkan hasil
print(f"Hasil pencarian:")
num_matched_documents = 0

for rank, idx in enumerate(top_document_indices, start=1):
    score = combined_scores[idx]
    if score > 0:  # Memeriksa apakah similarity lebih besar dari 0
        document_title = os.listdir(documents_folder)[idx]
        doc_id = document_title.split('_')[0]  # Dapatkan dokumen ID dari nama file
        print(f"Rank Dokumen: {rank}")
        print(f"Judul Dokumen: {document_title}")
        print(f"Similarity: {score}")
        num_matched_documents += 1
        print()

# Step 10: Menampilkan jumlah dokumen yang cocok dengan query
print(f"Jumlah dokumen yang cocok dengan query: {num_matched_documents}")
