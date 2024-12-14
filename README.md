# Lamogan-Travel-RS
Travel Recommendation System
Sistem Rekomendasi Tempat Wisata berbasis Machine Learning dan Natural Language Processing (NLP) untuk memberikan rekomendasi personal kepada pengguna berdasarkan preferensi mereka.
<br>
# 📌 Deskripsi
Lamogan-Travel-RS adalah sistem rekomendasi tempat wisata yang dirancang untuk membantu pengguna menemukan destinasi wisata terbaik. Aplikasi ini menggunakan kombinasi beberapa metode rekomendasi, termasuk Simple,  Content-Based Filtering, Collaborative Filtering, Collaborative Filtering with SVD. Selain itu, sistem ini menyediakan fitur visualisasi statistik dan daftar top 10 tempat terbaik.
<br>
# 📋 Fitur Utama
1. Sistem Rekomendasi
Menawarkan beberapa metode rekomendasi untuk pengalaman yang lebih personal:
- Simple Recommendation: Rekomendasi Simple.
- Content-Based Filtering: Rekomendasi berdasarkan fitur konten (nama, kategori, deskripsi tempat).
- Content-Based Filtering+ (TF-IDF): Versi lanjutan dengan penanganan term berbobot menggunakan TF-IDF.
- Collaborative Filtering (RecommenderNet): Menggunakan TensorFlow dan embedding untuk memprediksi preferensi berdasarkan data historis pengguna.
- Collaborative Filtering with SVD: Menggunakan teknik dekomposisi matriks untuk menangani dataset yang jarang (sparse).

2. Statistik Tempat Wisata
Menampilkan wawasan mendalam tentang data tempat wisata, seperti:
- Distribusi kategori tempat wisata.
- Distribusi rating tempat wisata.
- Kota dengan jumlah tempat wisata terbanyak.
- Hubungan harga dan rating tempat wisata.
- Statistik pengguna (usia, gender, lokasi).

3. Top 10 Tempat Wisata Terbaik
- Menghitung daftar tempat wisata terbaik berdasarkan Weighted Rating:
- Menggunakan rata-rata ulasan dan popularitas (jumlah ulasan).
- Mengkombinasikan skor dengan pendekatan persentil.
<br>
# 💡 Penjelasan Metode
1. Simple Recommendation
Metode paling dasar untuk rekomendasi dengan filter eksplisit:
- Input: Kategori, harga minimum/maksimum, rating minimum, jumlah ulasan.
- Output: Daftar tempat berdasarkan rating tertinggi yang sesuai filter.
- Kelebihan: Mudah digunakan, tidak memerlukan data historis pengguna.
- Kekurangan: Tidak personal, hasil filter sama untuk semua pengguna.

2. Content-Based Filtering
Rekomendasi berdasarkan kesamaan fitur konten (nama, kategori, deskripsi):
Algoritma:
- Menggabungkan fitur menjadi satu kolom (combined).
- Menggunakan CountVectorizer untuk representasi vektor fitur.
- Menghitung kesamaan menggunakan Cosine Similarity.
- Kelebihan: Tidak memerlukan data pengguna, fokus pada kesamaan konten.
- Kekurangan: Keterbatasan pada fitur yang disediakan.

3. Content-Based Filtering+ (TF-IDF)
Peningkatan metode content-based dengan pembobotan TF-IDF:
- TF-IDF memberikan bobot lebih tinggi untuk term unik dan menurunkan bobot untuk term umum.
- Membantu menangkap keunikan deskripsi tempat wisata, memberikan hasil yang lebih relevan.
- Kelebihan: Akurat untuk dataset dengan variasi deskripsi yang tinggi.
- Kekurangan: Tetap terbatas pada konten yang tersedia.

4. Collaborative Filtering (RecommenderNet)
Rekomendasi berdasarkan perilaku pengguna lain menggunakan TensorFlow:
Algoritma:
- Menggunakan embedding untuk merepresentasikan user dan tempat dalam ruang vektor.
- Model memprediksi rating untuk tempat yang belum dikunjungi.
- Menangkap pola laten dari preferensi pengguna.
- Kelebihan: Hasil personal yang mendalam.
- Kekurangan: Membutuhkan data historis pengguna (masalah cold-start).

5. Collaborative Filtering with SVD
Metode dekomposisi matriks untuk menangani sparsity data:
Algoritma:
- Membagi matriks interaksi user-item menjadi user latent dan item latent factors.
- Menggunakan kombinasi faktor untuk memprediksi rating.
- Kelebihan: Efektif untuk dataset yang sparsity tinggi.
- Kekurangan: Membutuhkan data cukup banyak untuk akurasi tinggi.
# 📊 Statistik dan Visualisasi
- Distribusi Kategori: Menampilkan jumlah tempat wisata per kategori.
- Distribusi Rating: Menunjukkan penyebaran rating tempat wisata dalam histogram.
- Kota Terbanyak: Kota dengan jumlah tempat wisata terbanyak.
- Hubungan Harga dan Rating: Scatterplot antara harga dan rating tempat wisata.
- Statistik Pengguna: Distribusi usia, gender, dan lokasi pengguna.
<br>
# 💻 Instalasi
- Clone repository ini:
git clone https://github.com/username/lamogan-travel-rs.git
cd lamogan-travel-rs
<br>
- Install dependencies:
pip install -r requirements.txt
- Jalankan aplikasi Streamlit:
<br>
streamlit run main3.py
# Techstack
- Python
- Streamlit (UI)
- TensorFlow (Collaborative Filtering)
- scikit-learn (Content-Based Filtering)
- pandas dan numpy (manipulasi data)
- Sastrawi (Text Preprocessing Bahasa Indonesia)
