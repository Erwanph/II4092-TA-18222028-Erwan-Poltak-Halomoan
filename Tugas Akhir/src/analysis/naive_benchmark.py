"""Pembanding naif (random walk) untuk menilai forecast skill model.

Metrik absolut (RMSE/MAPE/R2) tidak memberi tahu apakah model benar-benar
berguna pada deret yang sangat persisten seperti BI-Rate: galat kecil mudah
dicapai hanya dengan menebak "bulan depan sama dengan bulan ini". Modul ini
membandingkan setiap model terhadap tebakan naif tersebut melalui Theil's U2,
MASE, dan uji ketepatan arah Pesaran-Timmermann.

Catatan pustaka: uji Diebold-Mariano tetap dihitung sebagai diagnostik pendukung
dan tersimpan pada CSV, tetapi TIDAK dipakai pada laporan karena sumber rujukannya
tidak tersedia pada daftar pustaka yang sudah terverifikasi. Yang dilaporkan pada
buku adalah U2, MASE (keduanya bersandar pada Hyndman & Koehler 2006, yang juga
mendefinisikan U2 sebagai relative RMSE terhadap random walk untuk ramalan satu
langkah), serta uji Pesaran-Timmermann.

Sumber prediksi adalah results/experiments/analysis/pred_vs_actual_full.csv
(178 observasi Maret 2011-Desember 2025, artefak kanonis Windows), sehingga
angka di sini konsisten dengan registry tanpa perlu melatih ulang model.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PRED_PATH = Path("results/experiments/analysis/pred_vs_actual_full.csv")
OUT_CSV = Path("results/experiments/analysis/naive_benchmark.csv")
OUT_JSON = Path("results/experiments/analysis/naive_benchmark_summary.json")

ML_MODELS = ["RF", "BiLSTM", "LSTM", "XGBoost"]

# Model ekonometrika tidak menyimpan prediksi per bulan; U2-nya diturunkan dari
# RMSE uji pada registry (results/experiments/best_results_registry.json).
ECONOMETRIC_RMSE = {"ARIMA": 0.8974, "VAR": 2.7089}

STEP = 0.25  # BI-Rate hanya valid pada kelipatan 0,25%


def _rmse(e):
    return float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))


def _mae(e):
    return float(np.mean(np.abs(np.asarray(e, float))))


def pesaran_timmermann(y_dir, p_dir):
    """Uji nonparametrik Pesaran-Timmermann atas ketepatan arah prediksi.

    H0: arah aktual dan arah prediksi saling bebas, yakni ketepatan arah tidak
    lebih baik daripada kebetulan. Statistiknya asimtotik normal baku. Arah
    dikodekan biner (naik vs tidak-naik) karena BI-Rate memiliki tiga keadaan
    (naik, turun, tahan) sedangkan uji ini dirumuskan untuk dua keadaan.

    Dipakai menggantikan pembandingan berbasis galat kuadrat karena sumber
    rujukannya tersedia dan terverifikasi pada daftar pustaka penelitian ini.
    """
    y = (np.asarray(y_dir, float) > 0).astype(float)
    p = (np.asarray(p_dir, float) > 0).astype(float)
    n = y.size
    if n == 0:
        return {"PT_stat": float("nan"), "PT_p_value": float("nan")}

    P = float(np.mean(y == p))
    Py, Px = float(np.mean(y)), float(np.mean(p))
    Pstar = Py * Px + (1 - Py) * (1 - Px)

    var_P = Pstar * (1 - Pstar) / n
    var_Pstar = (((2 * Py - 1) ** 2) * Px * (1 - Px) / n
                 + ((2 * Px - 1) ** 2) * Py * (1 - Py) / n
                 + 4 * Py * Px * (1 - Py) * (1 - Px) / (n ** 2))
    denom = var_P - var_Pstar
    if denom <= 0:
        return {"PT_stat": float("nan"), "PT_p_value": float("nan")}

    stat = (P - Pstar) / np.sqrt(denom)
    p_val = 2 * (1 - stats.norm.cdf(abs(stat)))
    return {"PT_stat": float(stat), "PT_p_value": float(p_val)}


def diebold_mariano(e_model, e_naive, power=2, h=1):
    """Uji Diebold-Mariano dengan koreksi sampel kecil Harvey-Leybourne-Newbold.

    H0: kedua peramal memiliki akurasi yang sama (E[d]=0), dengan
    d_t = |e_model,t|^power - |e_naive,t|^power. Statistik negatif berarti model
    lebih baik daripada pembanding naif. Karena horizon peramalan h=1, tidak ada
    suku autokovarians yang perlu ditambahkan.
    """
    e_model = np.asarray(e_model, float)
    e_naive = np.asarray(e_naive, float)
    d = np.abs(e_model) ** power - np.abs(e_naive) ** power
    n = d.size
    d_bar = d.mean()

    gamma0 = np.sum((d - d_bar) ** 2) / n
    gamma = [np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / n for k in range(1, h)]
    v_d = (gamma0 + 2 * sum(gamma)) / n

    if v_d <= 0:
        return {"DM_stat": float("nan"), "DM_p_value": float("nan"),
                "mean_loss_diff": float(d_bar)}

    dm = d_bar / np.sqrt(v_d)
    # Koreksi HLN untuk sampel kecil, lalu bandingkan ke distribusi t(n-1).
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return {"DM_stat": float(dm_hln), "DM_p_value": float(p),
            "mean_loss_diff": float(d_bar)}


def load_predictions(path=PRED_PATH):
    """Muat prediksi dan lampirkan nilai aktual bulan sebelumnya (jangkar naif)."""
    df = pd.read_csv(path)
    df["aktual_prev"] = df["aktual"].shift(1)
    if df["aktual_prev"].isna().sum() != 1:
        raise ValueError("Deret aktual tidak kontinu; jangkar naif tidak sahih.")
    return df


def build_table(df):
    """Bandingkan setiap model terhadap random walk pada himpunan uji."""
    test = df[df["split"] == "test"].copy()
    train = df[df["split"] == "train"]

    y = test["aktual"].to_numpy(float)
    y_prev = test["aktual_prev"].to_numpy(float)
    e_rw = y - y_prev  # random walk: prediksi = nilai bulan sebelumnya

    # Penyebut MASE: MAE naif satu langkah pada data latih (Hyndman & Koehler).
    mae_naive_insample = float(np.mean(np.abs(np.diff(train["aktual"].to_numpy(float)))))

    is_change = np.round(np.abs(e_rw) / STEP) != 0
    is_hold = ~is_change

    rows = []
    rw_row = {
        "model": "Random Walk (naif)", "kelompok": "Pembanding",
        "RMSE": _rmse(e_rw), "MAE": _mae(e_rw),
        "MAPE": float(np.mean(np.abs(e_rw / y)) * 100),
        "Theil_U2": 1.0, "MASE": _mae(e_rw) / mae_naive_insample,
        "RMSE_change": _rmse(e_rw[is_change]), "RMSE_hold": _rmse(e_rw[is_hold]),
        "U2_change": 1.0, "U2_hold": float("nan"),
        "hit_rate_change": 0.0, "hit_rate_all": float(np.mean(is_hold)),
        "DM_stat": float("nan"), "DM_p_value": float("nan"),
        "PT_stat": float("nan"), "PT_p_value": float("nan"),
        "unggul_dari_naif": False,
    }
    rows.append(rw_row)

    for m in ML_MODELS:
        p = test[f"pred_{m}"].to_numpy(float)
        if np.isnan(p).any():
            raise ValueError(f"Prediksi {m} tidak lengkap pada himpunan uji.")
        e = y - p

        dir_true = np.sign(np.round((y - y_prev) / STEP))
        dir_pred = np.sign(np.round((p - y_prev) / STEP))

        dm = diebold_mariano(e, e_rw)
        pt = pesaran_timmermann(dir_true, dir_pred)
        u2 = _rmse(e) / _rmse(e_rw)
        rows.append({
            "model": m, "kelompok": "Machine Learning",
            "RMSE": _rmse(e), "MAE": _mae(e),
            "MAPE": float(np.mean(np.abs(e / y)) * 100),
            "Theil_U2": u2, "MASE": _mae(e) / mae_naive_insample,
            "RMSE_change": _rmse(e[is_change]), "RMSE_hold": _rmse(e[is_hold]),
            "U2_change": _rmse(e[is_change]) / _rmse(e_rw[is_change]),
            "U2_hold": float("nan"),  # galat naif nol pada bulan-tahan
            "hit_rate_change": float(np.mean(dir_true[is_change] == dir_pred[is_change])),
            "hit_rate_all": float(np.mean(dir_true == dir_pred)),
            "DM_stat": dm["DM_stat"], "DM_p_value": dm["DM_p_value"],
            "PT_stat": pt["PT_stat"], "PT_p_value": pt["PT_p_value"],
            "unggul_dari_naif": bool(u2 < 1),
        })

    for m, rmse in ECONOMETRIC_RMSE.items():
        rows.append({
            "model": m, "kelompok": "Ekonometrika",
            "RMSE": rmse, "MAE": float("nan"), "MAPE": float("nan"),
            "Theil_U2": rmse / rw_row["RMSE"], "MASE": float("nan"),
            "RMSE_change": float("nan"), "RMSE_hold": float("nan"),
            "U2_change": float("nan"), "U2_hold": float("nan"),
            "hit_rate_change": float("nan"), "hit_rate_all": float("nan"),
            "DM_stat": float("nan"), "DM_p_value": float("nan"),
            "PT_stat": float("nan"), "PT_p_value": float("nan"),
            "unggul_dari_naif": False,
        })

    meta = {
        "n_test": int(len(test)),
        "periode_uji": f"{test['tanggal'].iloc[0]} s.d. {test['tanggal'].iloc[-1]}",
        "n_change": int(is_change.sum()), "n_hold": int(is_hold.sum()),
        "rmse_random_walk": rw_row["RMSE"],
        "mae_naive_insample_train": mae_naive_insample,
        "catatan": ("U2 pada bulan-tahan tidak terdefinisi karena galat random "
                    "walk nol persis di sana."),
    }
    return pd.DataFrame(rows), meta


def report(out, meta):
    print("\n=== PEMBANDING NAIF (RANDOM WALK) PADA HIMPUNAN UJI ===")
    print(f"Periode uji: {meta['periode_uji']}  |  n={meta['n_test']} "
          f"({meta['n_change']} bulan-perubahan, {meta['n_hold']} bulan-tahan)")
    print(f"RMSE random walk: {meta['rmse_random_walk']:.4f}\n")
    hdr = (f"{'Model':<20}{'RMSE':>8}{'U2':>8}{'MASE':>7}{'U2_chg':>8}"
           f"{'hit_chg':>9}{'DM':>8}{'p':>8}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in out.iterrows():
        f = lambda v, s="7.4f": ("  n/a" if pd.isna(v) else format(v, s))
        print(f"{r.model:<20}{r.RMSE:>8.4f}{f(r.Theil_U2):>8}{f(r.MASE,'6.3f'):>7}"
              f"{f(r.U2_change,'7.4f'):>8}"
              f"{('  n/a' if pd.isna(r.hit_rate_change) else format(r.hit_rate_change,'8.1%')):>9}"
              f"{f(r.DM_stat,'7.3f'):>8}{f(r.DM_p_value,'7.3f'):>8}")

    print("\nTafsir: U2 > 1 berarti model KALAH dari tebakan naif secara "
          "keseluruhan.\n         U2_chg < 1 berarti model UNGGUL pada "
          "bulan-perubahan.\n         DM negatif + p < 0,05 berarti keunggulan "
          "model signifikan secara statistik.")


# Pembanding naif tidak lagi dilaporkan di buku TA; keluarannya dipertahankan
# sebagai diagnostik internal saja, sejalan dengan perlakuan atas uji
# Diebold-Mariano. Karena itu tabelnya ditulis ke folder hasil, bukan ke
# folder tables/ dokumen LaTeX.
TEX_PATH = Path("results/experiments/analysis/Tabel_Pembanding_Naif.tex")


def _tex_num(x, dec=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{dec}f}".replace(".", ",")


def _tex_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x*100:.0f}\\%"


def write_tex(out, meta, path=TEX_PATH):
    """Tabel Bab VI: kinerja tiap model terhadap ramalan naif."""
    lines = [
        "% Auto-generated oleh src/analysis/naive_benchmark.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Kinerja model terhadap pembanding naif pada himpunan uji "
        f"({meta['periode_uji']}, $n={meta['n_test']}$)}}",
        "\\label{tbl:pembanding-naif}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|r|r|r|r|c|c|r|}", "\\hline",
        "\\textbf{Model} & \\textbf{RMSE} & \\textbf{$U_2$} & \\textbf{MASE} & "
        "\\textbf{$U_2$\\textsubscript{ubah}} & \\textbf{Arah\\textsubscript{ubah}} & "
        "\\textbf{Arah\\textsubscript{total}} & \\textbf{$p$\\textsubscript{PT}} \\\\ \\hline",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"{r.model} & {_tex_num(r.RMSE)} & {_tex_num(r.Theil_U2, 3)} & "
            f"{_tex_num(r.MASE, 3)} & {_tex_num(r.U2_change, 3)} & "
            f"{_tex_pct(r.hit_rate_change)} & {_tex_pct(r.hit_rate_all)} & "
            f"{_tex_num(r.PT_p_value, 3)} \\\\")
        lines.append("\\hline")
    lines += [
        "\\end{tabular}%", "}",
        "\\par\\smallskip\\footnotesize $U_2$ dan MASE di bawah satu menandakan "
        "model mengungguli ramalan naif. Kolom $U_2$\\textsubscript{ubah} dihitung "
        f"hanya atas {meta['n_change']} bulan-perubahan; pada "
        f"{meta['n_hold']} bulan-tahan galat ramalan naif nol persis sehingga "
        "$U_2$ tidak terdefinisi. Kolom $p$\\textsubscript{PT} adalah nilai-$p$ uji "
        "Pesaran-Timmermann atas ketepatan arah terhadap peluang kebetulan.",
        "\\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def run(write=False):
    df = load_predictions()
    out, meta = build_table(df)
    report(out, meta)
    if write:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_CSV, index=False)
        pd.Series(meta).to_json(OUT_JSON, indent=2, force_ascii=False)
        write_tex(out, meta)
        print(f"\nDitulis: {OUT_CSV}\nDitulis: {OUT_JSON}")
    else:
        print("\nMode verifikasi saja (tanpa --write).")
    return out, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="tulis hasil ke CSV/JSON")
    run(write=ap.parse_args().write)
