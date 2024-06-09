import re
import os
import glob
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi Stemmer Sastrawi
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Fungsi untuk membersihkan teks
def clean_text(text):
    # Menghilangkan tanda kurung, tanda titik, tanda koma, tanda seru, dan angka
    text = re.sub(r'[-&()!./,0-9]', '', text)
    
    # Menghilangkan whitespace yang berlebihan dan menggantinya dengan spasi tunggal
    text = re.sub(r'\s+', ' ', text)
    
    # Menghilangkan stopwords
    stopwords = [
    "yang", "untuk", "pada", "ke", "para", "namun", "menurut", "antara", "dia", "dua", "ia", "seperti", "jika",
    "jika", "sehingga", "kembali", "dan", "tidak", "ini", "karena", "kepada", "oleh", "saat", "harus", "sementara",
    "setelah", "belum", "kami", "sekitar", "bagi", "serta", "di", "dari", "telah", "sebagai", "masih", "hal",
    "dalam", "bisa", "ada", "mereka", "akan", "saya", "atau", "kita", "dengan", "ini", "adalah", "itu", "sudah",
    "agar", "dari", "terhadap", "anda", "juga", "itu", "ya", "pada", "bahwa", "namun", "oleh", "karena",
    "sehingga", "kalau", "jika", "begitu", "apakah", "atas", "belum", "semua", "enggak", "daripada", "anu", "dst"
]

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    
    # Menggabungkan kata-kata yang tersisa kembali menjadi teks
    cleaned_text = ' '.join(filtered_words)
    
    return cleaned_text

# Fungsi untuk tokenisasi sederhana
def tokenize(text):
    # Memecah teks berdasarkan spasi
    tokens = text.split()
    return tokens

# Membuat folder untuk menyimpan hasil
output_folder = 'output_folder'
os.makedirs(output_folder, exist_ok=True)

# Mendapatkan daftar nama file dalam folder
input_folder = 'input_folder'
txt_files = glob.glob(os.path.join(input_folder, '*.txt'))

for input_file_name in txt_files:
    try:
        # Membaca teks dari berkas
        with open(input_file_name, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Membersihkan teks
        cleaned_text = clean_text(text)
        
        # Tokenisasi teks yang sudah dibersihkan
        tokens = tokenize(cleaned_text)
        
        # Stemming teks
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        
        # Menyimpan hasil teks yang sudah dibersihkan, di-tokenisasi, dan di-stem ke dalam berkas
        output_file_name = os.path.join(output_folder, os.path.basename(input_file_name))
        with open(output_file_name, 'w', encoding='utf-8') as output_file:
            output_file.write('\n'.join(stemmed_tokens))
        
        print(f"Hasil teks dari '{input_file_name}' telah disimpan ke dalam '{output_file_name}'.")
    
    except Exception as e:
        print(f"Terjadi kesalahan pada file '{input_file_name}': {e}")
