# Architectural Design Decisions (ADD) & Tech Trade-offs

Dokumen ini merinci keputusan desain arsitektur yang mendasari pengembangan **Indonesian News Search Engine**, mengevaluasi pilihan algoritma, skema pembobotan, serta trade-off teknologi untuk mendukung skalabilitas dan akurasi pencarian.

---

## 1. Normalisasi Skor & Pembobotan Kombinasi Hybrid

### Latar Belakang
Model pencarian Hybrid menggabungkan hasil dari model ruang vektor **TF-IDF** (berbasis Cosine Similarity) dan model probabilistik **BM25**. Namun, rentang skor (range) dari kedua algoritma ini sangat berbeda:
*   **TF-IDF (Cosine Similarity)**: Memiliki nilai batas yang pasti dalam interval $[0.0, 1.0]$.
*   **BM25**: Menggunakan perhitungan probabilistik yang tidak memiliki batas atas teoritis (skor berkisar dari $0$ hingga $+\infty$ tergantung pada panjang dokumen dan frekuensi kata).

Jika skor mentah langsung dijumlahkan tanpa normalisasi, skor BM25 akan mendominasi hasil akhir secara berlebihan, membuat kontribusi dari TF-IDF tidak signifikan.

### Keputusan Desain
Untuk mengatasi bias skala tersebut, sistem mengimplementasikan **Min-Max Normalization** untuk menyelaraskan skor dari masing-masing ranker ke rentang $[0.0, 1.0]$ sebelum dilakukan penggabungan linier tertimbang.

Persamaan normalisasi untuk suatu dokumen $d$ dalam daftar hasil pencarian $R$:
$$\text{Score}_{\text{normalized}}(d) = \frac{\text{Score}(d) - \min_{x \in R} \text{Score}(x)}{\max_{x \in R} \text{Score}(x) - \min_{x \in R} \text{Score}(x) + \epsilon}$$

Di mana $\epsilon = 1e-9$ ditambahkan untuk mencegah pembagian dengan nol ketika semua dokumen memiliki skor yang sama.

Skor gabungan akhir dihitung sebagai:
$$\text{Score}_{\text{hybrid}}(d) = w_{\text{tfidf}} \cdot \text{Score}_{\text{tfidf, normalized}}(d) + w_{\text{bm25}} \cdot \text{Score}_{\text{bm25, normalized}}(d)$$

Sistem menetapkan bobot default $w_{\text{tfidf}} = 0.5$ dan $w_{\text{bm25}} = 0.5$.

---

## 2. Perbandingan TF-IDF (Cosine Vector Space) vs. BM25 (Probabilistik)

| Dimensi | TF-IDF (Cosine Similarity) | BM25 (Okapi BM25) |
| :--- | :--- | :--- |
| **Dasar Teoretis** | Aljabar Linier & Geometri Vektor. Mengukur sudut antara vektor query dan vektor dokumen. | Teori Probabilitas Informasi. Mengukur probabilitas dokumen relevan dengan query. |
| **Saturasi Frekuensi Term (TF)** | Meningkat secara logaritmik tanpa batas atas: $\log(\text{tf} + 1)$. | Memiliki efek saturasi yang diatur oleh parameter $k_1$. Frekuensi kata yang sangat tinggi tidak menaikkan relevansi secara linier tanpa akhir. |
| **Normalisasi Panjang Dokumen** | Menggunakan normalisasi L2 (Cosine) yang membagi seluruh vektor dengan panjangnya. | Menggunakan normalisasi dinamis berbasis panjang dokumen rata-rata ($\text{avgdl}$) yang dikontrol oleh parameter $b$. |
| **Sensitivitas Dokumen Panjang** | Cenderung memberikan penalti besar pada dokumen panjang karena pembagian panjang vektor L2 secara global. | Memberikan kompensasi yang adil untuk dokumen panjang jika dokumen tersebut memiliki banyak variasi kata unik, bukan sekadar pengulangan kata kunci. |

### Mengapa Keduanya Digabungkan (Hybrid)?
*   **TF-IDF** sangat baik dalam mendeteksi kesamaan kosinus global dan memberikan bobot tinggi pada istilah langka secara konsisten.
*   **BM25** sangat kuat dalam mencocokkan dokumen berdasarkan pentingnya term secara lokal dengan mempertimbangkan saturasi frekuensi term dan panjang dokumen spesifik.
*   Hasil gabungan (Hybrid) memberikan keseimbangan antara kecocokan kata kunci yang presisi (BM25) dan relevansi kontekstual ruang vektor global (TF-IDF).

---

## 3. Analisis Trade-off: Inverted Index Tradisional vs. Vector Database (Fase 4 RAG)

Dalam rangka mempersiapkan sistem untuk **Fase 4 (Retrieval-Augmented Generation / RAG)**, berikut adalah perbandingan mendalam antara Inverted Index yang saat ini digunakan dengan Vector Database (Dense Retrieval):

| Fitur | Inverted Index (Lexical Search) | Vector Database (Semantic Search) |
| :--- | :--- | :--- |
| **Representasi Data** | Sparse Vectors / Posting Lists (Kamus kata ke ID dokumen). | Dense Embeddings (Vektor numerik dimensi tinggi, misal 768 atau 1536). |
| **Pencocokan** | Exact Keyword Matching (Kecocokan kata persis setelah stemming). | Semantic / Conceptual Similarity (Kecocokan makna/konteks walaupun menggunakan kata berbeda). |
| **Kebutuhan Memori** | Sangat Rendah (Indeks teks ringkas, mudah disimpan di disk/RAM kecil). | Tinggi (Memerlukan penyimpanan vektor float berdimensi tinggi di RAM untuk performa cepat). |
| **Kebutuhan Komputasi** | Sangat Rendah (Operasi pencarian set dan perkalian sparse sederhana). | Tinggi (Memerlukan perhitungan ANN - *Approximate Nearest Neighbor* seperti HNSW pada GPU/CPU). |
| **Penanganan Sinonim** | Terbatas (Memerlukan kamus sinonim manual atau spell correction tambahan). | Bawaan (Kata "vaksin" dan "imunisasi" berada dekat di ruang vektor embedding). |
| **Kasus Penggunaan Terbaik** | Pencarian kode produk, nama entitas unik, nomor seri, istilah teknis presisi. | Pencarian bermakna luas, kueri tanya-jawab natural, sistem rekomendasi konten. |

### Strategi Transisi ke RAG (Fase 4)
Untuk mempertahankan keunggulan kedua dunia, implementasi Fase 4 akan menggunakan pendekatan **Hybrid Search (Lexical + Semantic)**:
1.  **Lexical Search (BM25)**: Memastikan entitas penting, nama tempat, dan kata kunci spesifik tidak terlewatkan.
2.  **Semantic Search (Dense Embeddings + Vector DB)**: Mengambil dokumen berdasarkan pemahaman konteks pertanyaan pengguna secara alami.
3.  **Cross-Encoder Re-ranking**: Menggunakan model re-ranker kecil untuk mengurutkan ulang hasil gabungan teratas sebelum diumpankan ke Large Language Model (LLM) untuk sintesis jawaban akhir.
