import os
import re


# Fungsi untuk membersihkan dan tokenisasi teks
def clean_and_tokenize(text):
    text = re.sub(r'\n', ' ', text)  # Mengganti newline dengan spasi
    text = re.sub(r'\s+', ' ', text)  # Menghilangkan whitespace berlebihan
    tokens = text.split()
    return tokens

# Fungsi untuk membangun inverted index dari file
def build_inverted_index(file_path, doc_id):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Membersihkan dan tokenisasi teks
    tokens = clean_and_tokenize(text)
    
    # Membuat inverted index
    index = {}
    for i, token in enumerate(tokens):
        if token not in index:
            index[token] = []
        index[token].append((doc_id, i))
    
    return index

# Fungsi untuk mencari query dalam inverted index
def search_query(query, inverted_index, doc_names):
    results = []
    query_terms = query.split()
    for term in query_terms:
        if term in inverted_index:
            results.extend(inverted_index[term])
    return results

# Fungsi untuk menghitung jumlah query dalam dokumen
def count_queries_in_doc(doc_id, query_results):
    count = 0
    for result in query_results:
        if result[0] == doc_id:
            count += 1
    return count

# Membuat folder untuk menyimpan indeks
index_folder = 'index_folder'
os.makedirs(index_folder, exist_ok=True)

# Memproses setiap file dalam output_folder
output_folder = 'output_folder'
output_files = os.listdir(output_folder)

# Membuat daftar nama dokumen
doc_names = {}

for doc_id, output_file in enumerate(output_files):
    if output_file.endswith('.txt'):
        doc_names[doc_id] = output_file

# Membuat inverted index
inverted_index = {}

for doc_id, output_file in enumerate(output_files):
    if output_file.endswith('.txt'):
        file_path = os.path.join(output_folder, output_file)
        
        # Membuat inverted index untuk dokumen ini
        index = build_inverted_index(file_path, doc_id)
        
        # Memasukkan ke dalam inverted index global
        for term, positions in index.items():
            if term not in inverted_index:
                inverted_index[term] = []
            inverted_index[term].extend(positions)

# Menyimpan inverted index ke dalam berkas
index_file_path = os.path.join(index_folder, 'inverted_index.txt')
with open(index_file_path, 'w', encoding='utf-8') as index_file:
    for term, positions in inverted_index.items():
        index_file.write(f"{term}: {', '.join(map(str, positions))}\n")

print(f"Inverted index telah disimpan ke dalam '{index_file_path}'.")

# User memasukkan query
query = input("Masukkan query: ")

# Mencari query dalam inverted index
query_results = search_query(query, inverted_index, doc_names)

# Menghitung jumlah kemunculan query di setiap dokumen
doc_counts = {}
for doc_id in range(len(doc_names)):
    count = count_queries_in_doc(doc_id, query_results)
    if count > 0:
        doc_counts[doc_id] = count

# Mengurutkan dokumen berdasarkan jumlah kemunculan query
sorted_docs = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)

# Menampilkan hasil ranking
for doc_id, count in sorted_docs:
    positions = [result[1] for result in query_results if result[0] == doc_id]
    print(f"Kata '{query}' ditemukan di '{doc_names[doc_id]}' sebanyak {count} kali. pada posisi {positions}")
