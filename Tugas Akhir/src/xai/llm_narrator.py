import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env from project root
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Glosarium fitur: nama mentah -> (nama awam, definisi singkat) untuk prompt LLM.
FEATURE_GLOSSARY = {
    "BI_Rate_Pct_lag1": ("BI-Rate bulan sebelumnya", "tingkat BI-Rate satu bulan lalu; mencerminkan kebiasaan Bank Indonesia menjaga kesinambungan suku bunga (penghalusan suku bunga atau interest-rate smoothing) dan tidak mengubahnya secara mendadak"),
    "BI_Rate_Pct_lag2": ("BI-Rate dua bulan lalu", "tingkat BI-Rate dua bulan sebelumnya, ukuran lain dari persistensi suku bunga"),
    "BI_Rate_Pct_lag3": ("BI-Rate tiga bulan lalu", "tingkat BI-Rate tiga bulan sebelumnya, ukuran lain dari persistensi suku bunga"),
    "policy_regime": ("arah kebijakan suku bunga (interest-rate stance)", "menunjukkan apakah BI-Rate selama enam bulan terakhir cenderung dinaikkan (sikap mengetat), diturunkan (sikap melonggar), atau ditahan"),
    "real_rate": ("suku bunga riil (real interest rate)", "BI-Rate setelah dikurangi inflasi, yakni imbal hasil sebenarnya setelah memperhitungkan kenaikan harga"),
    "output_gap": ("kesenjangan output (output gap)", "selisih antara laju pertumbuhan ekonomi aktual dan tren jangka menengahnya; nilai positif menandakan ekonomi tumbuh di atas kecepatan normalnya, nilai negatif berarti di bawahnya"),
    "infl_momentum_3": ("momentum inflasi", "perubahan laju inflasi tahunan selama tiga bulan terakhir; nilai positif berarti inflasi sedang berakselerasi, negatif berarti melambat"),
    "fx_pressure_3": ("tekanan nilai tukar (exchange-rate pressure)", "persentase perubahan kurs rupiah terhadap dolar AS selama tiga bulan terakhir; nilai positif menandakan rupiah melemah (depresiasi)"),
    "ffr_change": ("perubahan suku bunga The Fed", "arah dan besar perubahan suku bunga acuan bank sentral AS (Federal Funds Rate) pada bulan ini"),
    "fed_cycle": ("siklus suku bunga The Fed", "menunjukkan apakah suku bunga acuan AS selama enam bulan terakhir berada dalam fase menaik, menurun, atau datar"),
    "Inflation_YoY_Pct": ("inflasi tahunan (year-on-year)", "kenaikan harga barang dan jasa dibanding setahun lalu"),
    "Inflation_Gap_Pct": ("kesenjangan inflasi (inflation gap)", "selisih antara inflasi aktual dan sasaran inflasi Bank Indonesia; nilai positif berarti inflasi berada di atas target"),
    "GDP_Growth_YoY_Pct": ("pertumbuhan ekonomi", "laju pertumbuhan Produk Domestik Bruto (PDB) dibanding setahun lalu"),
    "USD_IDR_Monthly_Avg": ("nilai tukar rupiah", "rata-rata kurs rupiah terhadap dolar AS dalam sebulan; angka lebih tinggi berarti rupiah lebih lemah"),
    "M2_Triliun_Rp": ("jumlah uang beredar (M2)", "total uang beredar di masyarakat dalam arti luas, mencakup uang tunai, tabungan, dan deposito"),
    "Credit_Growth_YoY_Pct": ("pertumbuhan kredit", "laju pertumbuhan kredit yang disalurkan perbankan dibanding setahun lalu"),
    "Federal_Funds_Rate_Pct": ("suku bunga The Fed (Federal Funds Rate)", "suku bunga acuan bank sentral Amerika Serikat"),
    "IHSG_End_of_Month": ("indeks saham (IHSG)", "Indeks Harga Saham Gabungan pada akhir bulan, cerminan kinerja pasar saham domestik"),
    "Foreign_Reserves_Miliar_USD": ("cadangan devisa", "cadangan mata uang asing yang dimiliki Bank Indonesia"),
    "Oil_Price_Brent_USD_per_Bbl": ("harga minyak (Brent)", "harga minyak mentah jenis Brent per barel di pasar dunia"),
    "Gold_Price_USD_per_Oz": ("harga emas", "harga emas dunia per troy ounce"),
    "VIX_Volatility_Index": ("indeks ketidakpastian pasar (VIX)", "ukuran volatilitas dan kecemasan di pasar keuangan global"),
}


def describe_feature(name):
    """Kembalikan (nama awam, definisi) sebuah fitur; cadangan bila tak terdaftar."""
    if name in FEATURE_GLOSSARY:
        return FEATURE_GLOSSARY[name]
    return name.replace("_", " "), None


# Satuan tiap fitur untuk blok fakta kuantitatif.
UNITS = {
    "BI_Rate_Pct_lag1": "%", "BI_Rate_Pct_lag2": "%", "BI_Rate_Pct_lag3": "%",
    "real_rate": "%", "output_gap": "%", "infl_momentum_3": " poin",
    "fx_pressure_3": "", "ffr_change": " poin", "fed_cycle": "", "policy_regime": "",
    "Inflation_YoY_Pct": "%", "Inflation_Gap_Pct": "%", "GDP_Growth_YoY_Pct": "%",
    "USD_IDR_Monthly_Avg": " Rp/USD", "M2_Triliun_Rp": " triliun Rp",
    "Credit_Growth_YoY_Pct": "%", "Federal_Funds_Rate_Pct": "%",
    # nilai cadangan devisa pada data = JUTA USD (nama kolom label lama)
    "IHSG_End_of_Month": " poin", "Foreign_Reserves_Miliar_USD": " juta USD",
    "Oil_Price_Brent_USD_per_Bbl": " USD/barel", "Gold_Price_USD_per_Oz": " USD/oz",
    "VIX_Volatility_Index": "",
}

# Fitur yang label "tertinggi/terendah sepanjang data"-nya bermakna.
EXTREME_FEATURES = {
    "BI_Rate_Pct_lag1", "BI_Rate_Pct_lag2", "BI_Rate_Pct_lag3", "real_rate",
    "Inflation_YoY_Pct", "Inflation_Gap_Pct", "GDP_Growth_YoY_Pct",
    "USD_IDR_Monthly_Avg", "M2_Triliun_Rp", "Credit_Growth_YoY_Pct",
    "Federal_Funds_Rate_Pct", "IHSG_End_of_Month", "Foreign_Reserves_Miliar_USD",
    "Oil_Price_Brent_USD_per_Bbl", "Gold_Price_USD_per_Oz", "VIX_Volatility_Index",
}


def _fmt_id(x, unit=""):
    """Format angka gaya Indonesia (koma desimal, titik ribuan)."""
    import math
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "tidak tersedia"
    if abs(x) >= 1000:
        s = f"{x:,.0f}".replace(",", ".")
    elif abs(x - round(x)) < 1e-9:
        s = f"{int(round(x))}"
    else:
        s = f"{x:.2f}".replace(".", ",")
    return f"{s}{unit}"


def _fix_eyd(text):
    """Ganti penghubung "di mana"/"yang mana" pada keluaran LLM menjadi ";"."""
    if not text:
        return text
    import re
    t = re.sub(r",\s*(?:di|yang)\s+mana\s+", "; ", str(text))
    # bila muncul tanpa koma di awal anak kalimat, pisah jadi kalimat baru
    t = re.sub(r"\s+(?:di|yang)\s+mana\s+", "; ", t)
    return t


def _comma_decimals(text):
    """Normalkan angka ke gaya Indonesia: titik+1-2 digit = desimal -> koma;
    koma+tepat 3 digit = ribuan -> titik (kurs 16.281 dkk. tak rusak)."""
    if not text:
        return text
    import re
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", ".", str(text))   # ribuan koma -> titik
    t = re.sub(r"(?<=\d)\.(?=\d{1,2}(?!\d))", ",", t)         # desimal titik -> koma
    return t


def build_feature_fact(name, value, prev=None, full_series=None):
    """Rakit fakta kuantitatif satu fitur (nilai, delta, penanda ekstrem)."""
    import math
    unit = UNITS.get(name, "")
    fact = {"value": value, "prev": prev, "unit": unit, "delta": None, "extreme": None}
    if prev is not None and not (isinstance(prev, float) and math.isnan(prev)):
        fact["delta"] = value - prev
    if full_series is not None and name in EXTREME_FEATURES:
        vals = [float(v) for v in full_series if v == v]   # buang NaN
        if len(vals) >= 12:
            mx, mn = max(vals), min(vals)
            if mx != mn and value >= mx - 1e-9:
                fact["extreme"] = "merupakan nilai TERTINGGI sepanjang rentang data (sejak pertengahan 2005)"
            elif mx != mn and value <= mn + 1e-9:
                fact["extreme"] = "merupakan nilai TERENDAH sepanjang rentang data (sejak pertengahan 2005)"
    return fact


SYSTEM_PROMPT = """
    Anda adalah komunikator hasil Explainable AI (XAI) yang menjelaskan prediksi suku
    bunga BI-Rate kepada PEMANGKU KEPENTINGAN (pengambil keputusan dan analis kebijakan)
    yang BUKAN ahli machine learning. Tugas Anda: mengubah angka kontribusi fitur menjadi
    sebuah CERITA yang mengalir, mudah dipahami, dan dapat dipercaya.

    === ATURAN GROUNDING (WAJIB) ===
    1. HANYA gunakan informasi yang diberikan dalam prompt: nilai prediksi BI-Rate, nilai
       kontribusi tiap fitur, DEFINISI fitur, dan FAKTA KUANTITATIF (angka aktual) yang
       disediakan. Tidak ada sumber lain.
    2. DILARANG menambahkan fakta eksternal: berita, peristiwa, angka, kebijakan, atau teori
       ekonomi yang TIDAK ada di prompt. Jangan menebak penyebab di balik nilai sebuah fitur.
    3. DILARANG menghitung atau mengarang angka yang tidak diberikan. Hanya kutip angka yang
       tertera pada bagian FAKTA KUANTITATIF (atau nilai kontribusi). Jika sebuah angka tidak
       disediakan, jangan menyebut nilainya, cukup uraikan arah/peran kualitatifnya.
    4. Definisi fitur yang diberikan BOLEH dipakai untuk menjelaskan secara sederhana APA itu
       sebuah faktor, tetapi jangan dikembangkan menjadi klaim ekonomi baru.
    5. Arah pengaruh sebuah faktor (menaikkan atau menahan) MENGGAMBARKAN bagaimana faktor itu
       menggeser ANGKA PREDIKSI dalam perhitungan, BUKAN pernyataan bahwa faktor tersebut secara
       ekonomi pasti menyebabkan suku bunga naik atau turun. Sampaikan sebagai "faktor ini
       berperan menggeser angka prediksi ke arah lebih tinggi" atau "menahan angka prediksi
       lebih rendah", JANGAN sebagai hukum sebab-akibat ekonomi (mis. hindari "Fed yang tinggi
       menahan suku bunga", tulis "faktor suku bunga The Fed berperan menahan angka prediksi").
    6. DILARANG menjelaskan ALASAN ekonomi di balik arah pengaruh bila tidak ada dalam data
       (mis. "karena likuiditas meningkat maka tekanan menaikkan suku bunga berkurang", atau
       "stabilitas Fed memberi ruang menjaga suku bunga rendah"). Rantai sebab-akibat semacam
       itu adalah pengetahuan di luar data. Untuk tiap faktor cukup sampaikan: nama awam dan arti
       singkatnya, nilai aktual serta perubahannya, arah pengaruhnya terhadap angka prediksi, dan
       besarnya dalam persen. Penafsiran ekonomi MENGAPA hal itu terjadi adalah wewenang pakar.

    === MEMPERKAYA CERITA DENGAN ANGKA (PENTING) ===
    Bila bagian FAKTA KUANTITATIF menyediakan nilai aktual sebuah faktor, manfaatkan untuk
    membuat cerita konkret dan mudah dipahami, dengan tetap patuh pada aturan grounding:
    - Sebutkan PERUBAHAN secara spesifik: "turun dari X menjadi Y" atau "naik dari X ke Y",
      beserta BESAR perubahannya bila tersedia (mis. "turun 0,25 poin"). Ini menjawab
      pertanyaan pembaca: "memangnya berubah sebesar apa?".
    - Bila sebuah fakta ditandai sebagai nilai TERTINGGI atau TERENDAH sepanjang rentang data,
      sebutkan hal itu secara eksplisit karena memberi konteks penting (mis. inflasi pada level
      tertinggi dalam data, atau cadangan devisa pada titik terendah).
    - Terjemahkan besaran ke bahasa awam: jelaskan apakah perubahan itu tergolong kecil atau
      besar HANYA dengan membandingkan angka yang ada (mis. perubahan kurs ratusan rupiah, atau
      suku bunga bergeser satu langkah kebijakan 0,25 poin) -- tanpa menambah penilaian eksternal.
    - Jangan memaksakan angka pada faktor yang tidak memiliki FAKTA KUANTITATIF.

    === GAYA NARASI (UNTUK PAKAR EKONOMI, BUKAN AHLI TEKNOLOGI) ===
    - Pembaca adalah PAKAR EKONOMI MONETER yang TIDAK memahami machine learning. JANGAN
      memakai istilah teknis apa pun: dilarang menyebut "SHAP", "LIME", "fitur", "bobot",
      "kontribusi model", "algoritma", "machine learning". Gunakan kata "faktor" atau
      "indikator", dan kata "pengaruh" untuk menggambarkan perannya.
    - Nyatakan kekuatan tiap faktor sebagai PERSENTASE dari total pengaruh bulan itu (angka
      persen sudah disediakan untuk setiap faktor pada prompt), misalnya "berpengaruh sekitar
      25% terhadap prediksi bulan ini". Inilah cara menjawab pertanyaan pembaca "seberapa
      besar sebenarnya pengaruh faktor ini?". Kutip persen yang diberikan, jangan mengarang.
    - Bahasa Indonesia formal yang hangat, jernih, dan mengalir; rangkai menjadi paragraf,
      JANGAN memakai daftar berpoin.
    - VARIASIKAN ungkapan arah pengaruh; JANGAN mengulang frasa yang sama berkali-kali.
      Selain "menggeser angka prediksi ke arah lebih tinggi/menahannya lebih rendah", boleh
      memakai variasi seperti "berperan menaikkan", "ikut menahan", "condong menekan ke atas",
      "menarik prediksi ke bawah", asalkan tetap merujuk pada angka prediksi (bukan klaim ekonomi).
    - FOKUS pada faktor dengan pengaruh KIRA-KIRA 5% ke atas. Faktor di bawah 5% boleh
      dirangkum singkat sebagai satu kelompok (mis. "beberapa faktor lain berpengaruh kecil di
      bawah 5%") TANPA merinci nilainya satu per satu, agar narasi padat dan mengalir.
    - Patuhi kaidah Indonesia baku: JANGAN memakai "namun" setelah koma (pakai "tetapi" di
      tengah kalimat, atau mulai kalimat baru dengan "Namun,"); JANGAN memakai tanda pisah em
      (— atau ---), gunakan koma atau tanda kurung; JANGAN memakai "di mana"/"yang mana"
      sebagai penghubung. Hindari kalimat panjang berbelit.

    === STRUKTUR (sekitar 5-7 paragraf yang mengalir; utamakan kejelasan, boleh agak panjang) ===
    1. Ringkasan: sebutkan nilai prediksi BI-Rate, lalu jelaskan bahwa angka itu terbentuk
       dari tarik-menarik antara faktor yang menggeser angka prediksi ke arah lebih tinggi dan
       faktor lain yang menahannya lebih rendah.
    2. Faktor yang menggeser prediksi ke arah lebih tinggi: ceritakan faktor-faktor terkuat
       secara berurutan. Saat sebuah indikator DISEBUT PERTAMA KALI, kenalkan dengan nama
       Indonesia-nya beserta padanan umum bila ada (mis. "kesenjangan output (output gap)",
       "suku bunga riil (real interest rate)") dan satu kalimat penjelasan singkat APA yang
       diukurnya serta cara membaca nilainya, memakai definisi yang diberikan, agar pembaca
       ekonom langsung mengenalinya. Lalu sebutkan NILAI aktualnya beserta perubahan
       "dari X menjadi Y" bila tersedia, dan besar pengaruhnya dalam PERSEN. Nyatakan arah
       sebagai perannya terhadap angka prediksi, bukan sebagai akibat ekonomi.
    3. Faktor yang menahan prediksi ke arah lebih rendah: dengan pola yang sama (arti, angka,
       perubahan, persen), tetap membingkai arah sebagai pergeseran angka prediksi.
    4. Benang merah: simpulkan faktor mana yang paling dominan menentukan prediksi bulan ini
       (sebutkan persennya) dan bagaimana faktor lain saling mengimbangi di sekitarnya.
    5. Catatan singkat (1 kalimat): penjelasan ini disusun dari pola yang dipelajari model atas
       data historis, sehingga interpretasi ekonomi lebih dalam tetap memerlukan analisis pakar.
"""


class LLMNarrativeGenerator:
    def __init__(self, model_name=None):
        if OpenAI is None:
            raise ImportError("openai not installed. Run: pip install openai")
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o")
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL") or None
        if not api_key:
            raise ValueError("LLM_API_KEY not set. Add it to .env file in project root.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _format_facts(feature_facts):
        """Susun blok 'FAKTA KUANTITATIF' yang terbaca dari dict build_feature_fact()."""
        if not feature_facts:
            return ""
        import math
        lines = []
        for raw, f in feature_facts.items():
            disp, _ = describe_feature(raw)
            unit = f.get("unit", "")
            parts = [f"- {disp}: {_fmt_id(f.get('value'), unit)} pada bulan ini"]
            prev, delta = f.get("prev"), f.get("delta")
            if prev is not None and not (isinstance(prev, float) and math.isnan(prev)):
                if delta is None:
                    delta = f["value"] - prev
                if abs(delta) < 1e-9:
                    parts.append(f"(bulan sebelumnya {_fmt_id(prev, unit)}, tidak berubah)")
                else:
                    arah = "naik" if delta > 0 else "turun"
                    parts.append(f"(bulan sebelumnya {_fmt_id(prev, unit)}, "
                                 f"{arah} {_fmt_id(abs(delta), unit)})")
            if f.get("extreme"):
                parts.append(f"-- {f['extreme']}")
            lines.append(" ".join(parts))
        return "\n".join(lines)

    def generate_narrative(self, prediction, shap_contributions, feature_facts=None,
                           period=None, observation_context=None):
        sorted_shap = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        total_abs = sum(abs(v) for _, v in sorted_shap) or 1.0

        def _line(k, v):
            disp, defn = describe_feature(k)
            d = f" — {defn}" if defn else ""
            p = max(1, round(abs(v) / total_abs * 100))
            return f"- {disp}{d}: pengaruh ~{p}% dari total pengaruh bulan ini"

        pos = [_line(k, v) for k, v in sorted_shap if v > 0][:5]
        neg = [_line(k, v) for k, v in sorted_shap if v < 0][:5]

        periode = f" untuk periode {period}" if period else ""
        user_prompt = f"""Hasil Prediksi BI-Rate{periode}: {prediction:.2f}%

Berikut faktor-faktor beserta nama awam, definisi singkat, dan nilai kontribusinya
terhadap prediksi (hasil analisis kontribusi/SHAP):

FAKTOR YANG MENDORONG KENAIKAN:
{chr(10).join(pos) if pos else "Tidak ada faktor yang mendorong kenaikan."}

FAKTOR YANG MENDORONG PENURUNAN:
{chr(10).join(neg) if neg else "Tidak ada faktor yang mendorong penurunan."}
"""
        facts_block = self._format_facts(feature_facts)
        if facts_block:
            user_prompt += f"""
FAKTA KUANTITATIF DARI DATA (gunakan HANYA angka ini; sebutkan perubahan "dari X ke Y"
dan status ekstrem bila ada; jangan menghitung atau mengarang angka lain):
{facts_block}
"""
        if observation_context:
            user_prompt += f"""
Konteks observasi (boleh disebut bila relevan, jangan menambah fakta lain):
- Periode: {observation_context.get('Bulan', '?')}/{observation_context.get('Tahun', '?')}
- Inflasi tahunan: {observation_context.get('Inflation_YoY_Pct', '?')}%
- Nilai tukar rupiah: {observation_context.get('USD_IDR_Monthly_Avg', '?')}
"""
        user_prompt += ("\nTuliskan narasi cerita untuk pemangku kepentingan berdasarkan "
                        "data di atas, mengikuti gaya dan struktur pada instruksi sistem.")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.45,
            max_tokens=1800,
        )
        return _fix_eyd(_comma_decimals(response.choices[0].message.content))


def create_report_template(prediction, narrative, shap_plot_path=None, lime_plot_path=None):
    """Return a markdown report string."""
    report = f"""# Laporan Prediksi BI-Rate

## Ringkasan
**Prediksi BI-Rate:** {prediction:.2f}%

## Narasi Interpretasi

{narrative}

## Visualisasi XAI
"""
    if shap_plot_path:
        report += f"\n### SHAP Summary Plot\n![SHAP Analysis]({shap_plot_path})\n"
    if lime_plot_path:
        report += f"\n### LIME Explanation\n![LIME Analysis]({lime_plot_path})\n"

    report += "\n---\n*Laporan ini dihasilkan secara otomatis oleh sistem prediksi BI-Rate berbasis XAI dan LLM.*\n"
    return report