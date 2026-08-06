"""Uji ablasi: kontribusi setiap kelompok fitur dan setiap komponen sistem.

Tiga jenis ablasi dijalankan.

1. Ablasi fitur - kelompok fitur dilepas satu per satu dari matriks masukan,
   sementara timeline, pembagian data, penskalaan, dan hyperparameter dibiarkan
   persis sama. Karena `apply_feature_engineering` menutup dengan `dropna()`,
   melepas fitur lewat konfigurasi eksperimen akan MENGUBAH panjang timeline dan
   membuat RMSE tidak sebanding; karena itu pelepasan dilakukan pada kolom
   matriks X setelah penyiapan, bukan pada konfigurasi.

2. Ablasi komponen - jangkar residual, ensembel tiga-seed, dan pembulatan ke
   grid kebijakan (snapping) dimatikan satu per satu.

3. Ablasi satu-per-satu (leave-one-out, `--only perfitur`) - tiap fitur dilepas
   sendirian agar sumbangan individualnya terbaca. Perlu diingat efeknya tidak
   aditif: jumlah delta per fitur tidak sama dengan delta ketika kelompoknya
   dilepas serentak, karena informasi antarfitur saling tumpang tindih.

Seluruh baris tabel dilatih ulang dengan seed yang sama, termasuk baris
konfigurasi penuh, agar selisih antarbaris mencerminkan efek ablasi dan bukan
selisih antara bobot beku dan pelatihan baru.

Catatan platform: RF, LSTM, dan BiLSTM reprodusibel lintas OS, sedangkan
XGBoost tidak (macOS 0,2207 pada konfigurasi penuh, Windows 0,1960). Sejak
31 Juli 2026 seluruh angka buku memakai lingkungan macOS. Selisih antarbaris
selalu sahih selama seluruh baris dihitung pada mesin yang sama.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.evaluation_core import prep, SPECS, ENSEMBLE_SEEDS, DISPLAY
from src.modeling.data_prep import create_sequences
from src.modeling.metrics import calculate_rmse, snap_to_grid
from src.modeling.machine_learning import RandomForestModel, XGBoostModel
from src.analysis.naive_benchmark import diebold_mariano

OUT_DIR = Path("results/experiments/analysis")
TEX_FITUR = Path("../Tugas Akhir - Dokumen/tables/Tabel_Ablasi_Fitur.tex")
TEX_KOMPONEN = Path("../Tugas Akhir - Dokumen/tables/Tabel_Ablasi_Komponen.tex")
TEX_PERFITUR = Path("../Tugas Akhir - Dokumen/tables/Tabel_Ablasi_PerFitur.tex")

STEP = 0.25
MODELS = ["RF", "BiLSTM", "LSTM", "XGBoost"]

LAGS = ["BI_Rate_Pct_lag1", "BI_Rate_Pct_lag2", "BI_Rate_Pct_lag3"]
TURUNAN = ["policy_regime", "real_rate", "output_gap", "infl_momentum_3",
           "fx_pressure_3", "ffr_change", "fed_cycle"]

# label -> fungsi yang mengembalikan daftar fitur yang DIBUANG
VARIAN_FITUR = [
    ("Konfigurasi penuh", lambda f: []),
    ("Tanpa BI-Rate $t-1$", lambda f: ["BI_Rate_Pct_lag1"]),
    ("Tanpa seluruh \\textit{lag} BI-Rate", lambda f: LAGS),
    ("Tanpa fitur turunan reaksi kebijakan", lambda f: TURUNAN),
    ("Tanpa indikator makro mentah",
     lambda f: [c for c in f if c not in LAGS + TURUNAN]),
    ("Hanya \\textit{lag} BI-Rate", lambda f: [c for c in f if c not in LAGS]),
]


def _rw_reference(d):
    """RMSE random walk pada himpunan uji + penanda bulan-perubahan."""
    y, split = d["y"], d["split"]
    idx = np.where(split == "test")[0]
    y_te = y[idx]
    y_prev = y[idx - 1]
    e_rw = y_te - y_prev
    is_change = np.round(np.abs(e_rw) / STEP) != 0
    rms = lambda e: float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))
    return {
        "rmse_rw": rms(e_rw),
        "rmse_rw_change": rms(e_rw[is_change]),
        "is_change": is_change, "idx": idx, "y_te": y_te, "y_prev": y_prev,
    }


def _score(ref, pred_full, snap=True):
    """Metrik uji sebuah varian.

    RMSE selalu dihitung dari prediksi MENTAH (sesuai konvensi Bab VI). Argumen
    `snap` hanya memengaruhi metrik pelaporan kebijakan: ketepatan nilai persis
    pada grid 0,25%. Akurasi arah tidak dipengaruhi snapping karena selisih
    terhadap bulan sebelumnya sudah dibulatkan ke grid saat tanda diambil.
    """
    idx, y_te, y_prev = ref["idx"], ref["y_te"], ref["y_prev"]
    chg = ref["is_change"]
    p = np.asarray(pred_full, float)[idx]
    ok = ~np.isnan(p)
    if not ok.all():                      # sequence_length memotong awal timeline
        y_te, y_prev, chg, p = y_te[ok], y_prev[ok], chg[ok], p[ok]

    rmse = calculate_rmse(y_te, p)
    p_lapor = snap_to_grid(p, STEP) if snap else p
    dir_true = np.sign(np.round((y_te - y_prev) / STEP))
    dir_pred = np.sign(np.round((p_lapor - y_prev) / STEP))

    e_rw = y_te - y_prev
    dm = diebold_mariano(y_te - p, e_rw)
    return {
        "RMSE_test": rmse,
        "Theil_U2": rmse / ref["rmse_rw"],
        "RMSE_change": calculate_rmse(y_te[chg], p[chg]) if chg.any() else np.nan,
        "U2_change": (calculate_rmse(y_te[chg], p[chg]) / ref["rmse_rw_change"]
                      if chg.any() else np.nan),
        "hit_change": (float(np.mean(dir_true[chg] == dir_pred[chg]))
                       if chg.any() else np.nan),
        "hit_all": float(np.mean(dir_true == dir_pred)),
        # ketepatan nilai persis pada grid kebijakan: inilah yang dijamin snapping
        "exact_hit": float(np.mean(np.abs(p_lapor - y_te) < 1e-9)),
        "DM_stat": dm["DM_stat"], "DM_p_value": dm["DM_p_value"],
    }


def _fit_tree(name, d, keep, seed=42):
    X, y = d["X"][:, keep], d["y"]
    vi, ti = d["val_idx"], d["test_idx"]
    p = dict(SPECS[name]["p"])
    if name == "RF":
        m = RandomForestModel(**p)
        m.fit(X[:vi], y[:vi])
    else:
        m = XGBoostModel(**p)
        m.fit(X[:vi], y[:vi], X[vi:ti], y[vi:ti])
    return m.predict(X)


def _fit_dl(name, d, keep, residual=True, seeds=ENSEMBLE_SEEDS):
    from src.modeling.deep_learning import LSTMModel, BiLSTMModel
    X, y = d["X"][:, keep], d["y"]
    vi = d["val_idx"]
    p = SPECS[name]["p"]
    seq = p["sequence_length"]
    kind = SPECS[name]["kind"]

    Xseq, _ = create_sequences(X, y, seq)
    anc = y[seq - 1:-1]
    Xtr_s, ytr_s = create_sequences(X[:vi], y[:vi], seq)
    anc_tr = np.asarray(y[:vi], float)[seq - 1:-1]
    Cls = LSTMModel if kind == "LSTM" else BiLSTMModel

    preds = []
    for s in seeds:
        m = Cls(sequence_length=seq, n_features=Xseq.shape[2], units=p["units"],
                dropout=p["dropout"], learning_rate=p["learning_rate"],
                batch_size=p["batch_size"], epochs=p["epochs"],
                residual=residual, seed=s)
        m.fit(Xtr_s, ytr_s, anchor_train=anc_tr)
        out = np.full(len(y), np.nan)
        out[seq:] = m.predict(Xseq, anchor=anc)
        preds.append(out)
    return np.nanmean(preds, axis=0)


def _predict(name, d, keep, residual=True, seeds=ENSEMBLE_SEEDS):
    if SPECS[name]["kind"] in ("RF", "XGBoost"):
        return _fit_tree(name, d, keep)
    return _fit_dl(name, d, keep, residual=residual, seeds=seeds)


def run_feature_ablation(models=MODELS):
    rows = []
    for name in models:
        d = prep(SPECS[name]["exp"])
        ref = _rw_reference(d)
        feats = d["feats"]
        base_rmse = None
        for label, dropper in VARIAN_FITUR:
            drop = set(dropper(feats)) & set(feats)
            keep = [i for i, f in enumerate(feats) if f not in drop]
            if not keep:
                continue
            sc = _score(ref, _predict(name, d, keep))
            if base_rmse is None:
                base_rmse = sc["RMSE_test"]
            rows.append({
                "model": name, "varian": label,
                "n_fitur": len(keep), "n_dibuang": len(drop),
                **sc, "delta_RMSE": sc["RMSE_test"] - base_rmse,
            })
            print(f"[{name:7s}] {label:<38s} n={len(keep):2d} "
                  f"RMSE={sc['RMSE_test']:.4f} (Δ{sc['RMSE_test']-base_rmse:+.4f}) "
                  f"U2={sc['Theil_U2']:.3f} U2chg={sc['U2_change']:.3f} "
                  f"hit_chg={sc['hit_change']:.0%} p_DM={sc['DM_p_value']:.3f}")
    return pd.DataFrame(rows)


def run_component_ablation():
    """Ablasi jangkar residual, ensembel, dan snapping."""
    rows = []
    for name in ["BiLSTM", "LSTM"]:
        d = prep(SPECS[name]["exp"])
        ref = _rw_reference(d)
        keep = list(range(d["X"].shape[1]))

        full = _predict(name, d, keep)
        base = _score(ref, full)
        variants = [
            ("Konfigurasi penuh", base),
            ("Tanpa pembulatan ke grid kebijakan", _score(ref, full, snap=False)),
            ("Tanpa ensembel (1 \\textit{seed})",
             _score(ref, _predict(name, d, keep, seeds=[42]))),
            ("Tanpa jangkar residual",
             _score(ref, _predict(name, d, keep, residual=False))),
        ]
        for label, sc in variants:
            rows.append({"model": name, "varian": label, **sc,
                         "delta_RMSE": sc["RMSE_test"] - base["RMSE_test"]})
            print(f"[{name:7s}] {label:<38s} RMSE={sc['RMSE_test']:.4f} "
                  f"(Δ{sc['RMSE_test']-base['RMSE_test']:+.4f}) "
                  f"U2={sc['Theil_U2']:.3f} hit_chg={sc['hit_change']:.0%} "
                  f"tepat_grid={sc['exact_hit']:.0%}")

    for name in ["RF", "XGBoost"]:
        d = prep(SPECS[name]["exp"])
        ref = _rw_reference(d)
        keep = list(range(d["X"].shape[1]))
        full = _predict(name, d, keep)
        base = _score(ref, full)
        for label, sc in [("Konfigurasi penuh", base),
                          ("Tanpa pembulatan ke grid kebijakan",
                           _score(ref, full, snap=False))]:
            rows.append({"model": name, "varian": label, **sc,
                         "delta_RMSE": sc["RMSE_test"] - base["RMSE_test"]})
            print(f"[{name:7s}] {label:<38s} RMSE={sc['RMSE_test']:.4f} "
                  f"hit_chg={sc['hit_change']:.0%}")
    return pd.DataFrame(rows)


def _num(x, dec=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{dec}f}".replace(".", ",").replace("-", "$-$")


def _pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x*100:.0f}\\%"


def write_feature_tex(df, path=TEX_FITUR):
    lines = [
        "% Auto-generated oleh src/analysis/ablation_study.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Kinerja tiap varian masukan pada ablasi kelompok fitur}",
        "\\label{tbl:ablasi-fitur}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|l|c|r|r|r|c|}", "\\hline",
        "\\textbf{Model} & \\textbf{Varian} & \\textbf{Fitur} & \\textbf{RMSE} & "
        "$\\Delta$\\textbf{RMSE} & \\textbf{RMSE\\textsubscript{ubah}} & "
        "\\textbf{Arah\\textsubscript{ubah}} \\\\ \\hline",
    ]
    for model, grp in df.groupby("model", sort=False):
        n = len(grp)
        for k, (_, r) in enumerate(grp.iterrows()):
            first = (f"\\multirow{{{n}}}{{*}}{{{model}}}" if k == 0 else "")
            lines.append(
                f"{first} & {r.varian} & {int(r.n_fitur)} & {_num(r.RMSE_test)} & "
                f"{_num(r.delta_RMSE)} & {_num(r.RMSE_change)} & "
                f"{_pct(r.hit_change)} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}%", "}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def write_component_tex(df, path=TEX_KOMPONEN):
    lines = [
        "% Auto-generated oleh src/analysis/ablation_study.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Uji ablasi komponen sistem}",
        "\\label{tbl:ablasi-komponen}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|l|r|r|c|c|}", "\\hline",
        "\\textbf{Model} & \\textbf{Komponen dimatikan} & \\textbf{RMSE} & "
        "$\\Delta$\\textbf{RMSE} & \\textbf{Arah\\textsubscript{ubah}} & "
        "\\textbf{Tepat grid} \\\\ \\hline",
    ]
    for model, grp in df.groupby("model", sort=False):
        n = len(grp)
        for k, (_, r) in enumerate(grp.iterrows()):
            first = (f"\\multirow{{{n}}}{{*}}{{{model}}}" if k == 0 else "")
            lines.append(
                f"{first} & {r.varian} & {_num(r.RMSE_test)} & {_num(r.delta_RMSE)} & "
                f"{_pct(r.hit_change)} & {_pct(r.exact_hit)} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}%", "}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def run_perfeature_ablation(models=MODELS):
    """Ablasi satu-per-satu (leave-one-out): tiap fitur dilepas sendirian.

    Melengkapi ablasi kelompok, yang hanya menjawab peran gugus fitur. Di sini
    pertanyaannya lebih halus: seberapa besar sumbangan tiap fitur secara
    individual terhadap galat uji. Nilai delta positif berarti melepas fitur
    memperburuk prediksi, jadi fitur itu memang dipakai model.
    """
    rows = []
    for name in models:
        d = prep(SPECS[name]["exp"])
        ref = _rw_reference(d)
        feats = d["feats"]
        semua = list(range(len(feats)))
        base = _score(ref, _predict(name, d, semua))
        rows.append({"model": name, "fitur": "(konfigurasi penuh)",
                     "fitur_tampil": "(konfigurasi penuh)", **base, "delta_RMSE": 0.0})
        print(f"[{name:7s}] konfigurasi penuh          RMSE={base['RMSE_test']:.4f}")
        for i, f in enumerate(feats):
            keep = [j for j in semua if j != i]
            sc = _score(ref, _predict(name, d, keep))
            delta = sc["RMSE_test"] - base["RMSE_test"]
            rows.append({"model": name, "fitur": f,
                         "fitur_tampil": DISPLAY.get(f, f), **sc, "delta_RMSE": delta})
            print(f"[{name:7s}] tanpa {f:<24s} RMSE={sc['RMSE_test']:.4f} "
                  f"(Δ{delta:+.4f}) U2chg={sc['U2_change']:.3f}")
    return pd.DataFrame(rows)


def write_perfeature_tex(df, path=TEX_PERFITUR):
    """Baris = fitur, kolom = RMSE uji tiap model (hanya fitur yang dipakai semua).

    Nilainya ditulis apa adanya, bukan selisih terhadap konfigurasi penuh, agar
    besaran galat tiap varian terbaca langsung. Baris pertama memuat konfigurasi
    penuh sebagai pembanding, dan urutan baris tetap mengikuti besar selisihnya.
    """
    inti = df[df.fitur != "(konfigurasi penuh)"]
    bersama = sorted(set.intersection(
        *(set(g.fitur) for _, g in inti.groupby("model"))))
    urut_model = [m for m in ["RF", "XGBoost", "BiLSTM", "LSTM"]
                  if m in set(inti.model)]
    penuh = df[df.fitur == "(konfigurasi penuh)"].set_index("model")
    piv = inti.pivot_table(index="fitur", columns="model", values="RMSE_test")
    piv = piv.loc[bersama, urut_model]
    tampil = dict(zip(inti.fitur, inti.fitur_tampil))
    piv["rata"] = piv.mean(axis=1)

    # Galat yang sama, tetapi diukur khusus pada bulan-perubahan. Sebagian fitur
    # makro tidak terbaca pada RMSE agregat dan baru terlihat di sini.
    piv_ubah = inti.pivot_table(index="fitur", columns="model",
                                values="RMSE_change").loc[bersama, urut_model]
    piv["rata_ubah"] = piv_ubah.mean(axis=1)
    piv = piv.sort_values("rata", ascending=False)

    dasar = {m: penuh.loc[m, "RMSE_test"] for m in urut_model}
    dasar_ubah = {m: penuh.loc[m, "RMSE_change"] for m in urut_model}
    baris_dasar = (
        "(konfigurasi penuh) & "
        + " & ".join(_num(dasar[m]) for m in urut_model)
        + " & " + _num(sum(dasar.values()) / len(dasar))
        + " & " + _num(sum(dasar_ubah.values()) / len(dasar_ubah)) + " \\\\")

    lines = [
        "% Auto-generated oleh src/analysis/ablation_study.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{RMSE uji tiap model ketika satu fitur dilepas sendirian}",
        "\\label{tbl:ablasi-perfitur}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|" + "r|" * (len(urut_model) + 2) + "}", "\\hline",
        "\\textbf{Fitur dilepas} & "
        + " & ".join(f"\\textbf{{{m}}}" for m in urut_model)
        + " & \\textbf{rata-rata} & "
          "\\textbf{rata-rata\\textsubscript{ubah}} \\\\ \\hline",
        baris_dasar, "\\hline", "\\hline",
    ]
    for f, r in piv.iterrows():
        lines.append(f"{tampil.get(f, f)} & "
                     + " & ".join(_num(r[m]) for m in urut_model)
                     + f" & {_num(r['rata'])} & {_num(r['rata_ubah'])} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}%", "}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="tulis CSV dan tabel LaTeX")
    ap.add_argument("--only", choices=["fitur", "komponen", "perfitur"],
                    help="jalankan satu jenis saja")
    ap.add_argument("--models", nargs="+", default=MODELS,
                    help="batasi model (mis. --models RF XGBoost)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.only == "perfitur":
        print("\n=== ABLASI SATU-PER-SATU (LEAVE-ONE-OUT) ===")
        per = run_perfeature_ablation(args.models)
        if args.write:
            per.to_csv(OUT_DIR / "ablation_per_feature.csv", index=False)
            write_perfeature_tex(per)
            print(f"CSV: {OUT_DIR/'ablation_per_feature.csv'}")
        return

    if args.only != "komponen":
        print("\n=== ABLASI FITUR ===")
        fit = run_feature_ablation()
        if args.write:
            fit.to_csv(OUT_DIR / "ablation_features.csv", index=False)
            write_feature_tex(fit)
            print(f"CSV: {OUT_DIR/'ablation_features.csv'}")

    if args.only != "fitur":
        print("\n=== ABLASI KOMPONEN ===")
        comp = run_component_ablation()
        if args.write:
            comp.to_csv(OUT_DIR / "ablation_components.csv", index=False)
            write_component_tex(comp)
            print(f"CSV: {OUT_DIR/'ablation_components.csv'}")


if __name__ == "__main__":
    main()
