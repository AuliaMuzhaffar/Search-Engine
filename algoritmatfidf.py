import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# Membaca file urldoc.json
with open('urldoc.json', 'r', encoding='utf-8') as json_file:
    urldoc = json.load(json_file)

# Membuat vektor TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# User memasukkan query
query = input("Masukkan query: ")

# Mengubah query menjadi vektor TF-IDF
query_vector = tfidf_vectorizer.transform([query])

# Menghitung skor kesamaan (cosine similarity) antara query dan dokumen
similarities = cosine_similarity(query_vector, tfidf_matrix)

# Mendapatkan indeks dokumen dengan skor tertinggi
top_document_indices = similarities.argsort()[0][::-1]  # Ambil semua dokumen terurut

# Menampilkan hasil
print(f"Masukkan query: {query}")
print(f"Hasil pencarian dari TF-IDF:")
num_results = 0  # Variabel untuk menghitung jumlah dokumen yang cocok dengan query
for idx in top_document_indices:
    similarity_score = similarities[0][idx]
    if similarity_score > 0:  # Menampilkan hanya jika similarity lebih besar dari 0
        num_results += 1
        document_title = os.listdir(documents_folder)[idx]
        url = urldoc.get(document_title, "URL tidak tersedia")  # Mendapatkan URL dari urldoc.json
        print(f"Rank : {num_results}")
        print(f"Judul Dokumen: {document_title}")
        print(f"Similarity: {similarity_score}")
        print(f"Link URL: {url}")
        print()

print(f"Jumlah dokumen yang cocok dengan query adalah: {num_results} dokumen")
