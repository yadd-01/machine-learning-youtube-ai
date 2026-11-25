"""Daftar kata yang dilindungi dalam preprocessing dan analisis noise.

Dipakai oleh:
    - preprocessing/preprocessing.py  -> tidak dihapus stopword, tidak di-stem
    - analysis/noise_analysis.py      -> tidak dihitung sebagai kata berulang

Alasan perlindungan:
    1. Istilah inti topik AI tidak boleh hilang dari WordCloud.
    2. Beberapa kata rusak maknanya jika di-stem Sastrawi
       (mis. "pemerintah" -> "perintah").
"""

PROTECTED_WORDS = frozenset(
    {
        # Istilah inti topik
        "ai",
        "artificial",
        "intelligence",
        "artifisial",
        "kecerdasan",
        "buatan",
        "machine",
        "learning",
        "deep",
        "neural",
        "network",
        "nlp",
        "llm",
        "gpt",
        "chatgpt",
        "openai",
        "gemini",
        "claude",
        "copilot",
        "chatbot",
        "robot",
        "teknologi",
        "digital",
        "komputer",
        "algoritma",
        "otomasi",
        "data",
        "model",
        "internet",
        "aplikasi",
        "sistem",
        "informasi",
        "generatif",
        # Pengecualian stemming (Sastrawi merusak makna kata ini)
        "pemerintah",
        "pemerintahan",
        "belajar",
    }
)
