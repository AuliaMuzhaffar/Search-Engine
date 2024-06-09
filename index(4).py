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

# Membuat folder untuk menyimpan indeks
index_folder = 'index_folder'
os.makedirs(index_folder, exist_ok=True)

# Memproses setiap file dalam output_folder
output_folder = 'output_folder'
output_files = os.listdir(output_folder)

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
