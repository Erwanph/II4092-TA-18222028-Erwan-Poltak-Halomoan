"""Gambar interpretasi XAI per model per bulan evaluasi (SHAP, LIME, waterfall).

Menghasilkan PNG di results/experiments/evaluasi/xai_charts/ dengan pola nama
<Model>_<Bulan>_{shap|lime|waterfall}.png. Gambar inilah yang disisipkan penulis
secara manual ke lembar evaluasi domain expert dan lampiran laporan.
BERAT -- jalankan manual: python -m src.generate_xai_charts
"""

from pathlib import Path

from src.evaluation_core import (
    SPECS, MODEL_ORDER, EVAL_DIR,
    prep, train_predict, metrics_test, pick_control_months,
    shap_explanation, lime_instance, _disp, _f2, _f4,
)

CHART_DIR = EVAL_DIR / "xai_charts"


def _contrib_bar(contrib, judul, save_path, topn=8):
    """Grafik batang horizontal kontribusi per faktor untuk SATU observasi.

    Dipakai seragam untuk SHAP maupun LIME (keduanya menghasilkan dict
    {fitur: kontribusi}). Merah = mendorong prediksi naik, biru = menahan turun.
    Mengembalikan path bila berhasil, None bila tak ada data/gagal.
    """
    if not contrib:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:topn]
        items = items[::-1]                       # kontribusi terbesar di atas
        labels = [_disp(k) for k, _ in items]
        vals = [float(v) for _, v in items]
        colors = ["#C0504D" if v > 0 else "#4472C4" for v in vals]
        fig, ax = plt.subplots(figsize=(5.8, 3.3))
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="#7F7F7F", lw=0.8)
        ax.set_title(judul, fontsize=9, fontweight="bold")
        ax.set_xlabel("kontribusi terhadap prediksi  (merah: mendorong naik, biru: menahan turun)",
                      fontsize=6.5)
        ax.tick_params(axis="x", labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return save_path
    except Exception as e:
        print(f"  [grafik] dilewati ({judul}): {e}")
        return None


def _shap_waterfall(expl_obj, judul, save_path, max_display=10):
    """Render SHAP ``waterfall plot`` asli (shap.plots.waterfall) untuk satu observasi.

    Waterfall menampilkan bagaimana prediksi terbentuk secara aditif dari nilai
    dasar (rata-rata keluaran model) lalu didorong naik/turun oleh tiap faktor
    hingga mencapai angka akhir. Mengembalikan path bila berhasil, None bila gagal
    (fallback: cukup pakai grafik batang kontribusi)."""
    if expl_obj is None:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        plt.figure()
        shap.plots.waterfall(expl_obj, max_display=max_display, show=False)
        fig = plt.gcf()
        fig.set_size_inches(6.4, 3.8)
        fig.suptitle(judul, fontsize=9, fontweight="bold")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return save_path
    except Exception as e:
        print(f"  [waterfall] dilewati ({judul}): {e}")
        return None
    # d0 = trained[MODEL_ORDER[0]]["d"]
    # i = int(np.where(d0["dates"] == "2026-Mei")[0][0])
    # controls = [{"i": i, "label": str(d0["dates"][i]), "kategori": "khusus", "prev": float(d0["y"][i - 1]), "actual": float(d0["y"][i])}]


def main():
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Latih seluruh model dahulu agar bulan kontrol dapat dipilih lintas-model.
    trained = {}
    for name in MODEL_ORDER:
        print(f"\n=== Latih {name} ===")
        d = prep(SPECS[name]["exp"])
        pred_raw, pred_snap, art = train_predict(name, d)
        met = metrics_test(d, pred_snap, pred_raw)
        print(f"  Uji: RMSE={_f4(met['RMSE'])} MAPE={_f2(met['MAPE'])}% R2={_f4(met['R2'])}")
        trained[name] = {"d": d, "pred_snap": pred_snap, "art": art, "met": met}

    # 2) Dua bulan kontrol global (sama untuk semua model).
    for c in controls:
        print(f"Bulan kontrol [{c['kategori']}]: {c['label']} (aktual {c['actual']})")

    # 3) Tiga gambar per (model, bulan): batang SHAP, batang LIME, waterfall SHAP.
    for name in MODEL_ORDER:
        t = trained[name]
        for c in controls:
            i = c["i"]
            tag = f"{name}_{c['label']}"
            contrib, shap_expl = shap_explanation(name, t["d"], t["art"], i)
            lime_contrib = lime_instance(name, t["d"], t["art"], i)
            _contrib_bar(contrib, f"Kontribusi SHAP - {name} ({c['label']})",
                         str(CHART_DIR / f"{tag}_shap.png"))
            _contrib_bar(lime_contrib, f"Kontribusi LIME - {name} ({c['label']})",
                         str(CHART_DIR / f"{tag}_lime.png"))
            _shap_waterfall(shap_expl, f"SHAP Waterfall - {name} ({c['label']})",
                            str(CHART_DIR / f"{tag}_waterfall.png"))
            print(f"  -> {tag}_{{shap,lime,waterfall}}.png")
    print(f"\nSelesai. Gambar XAI per model/bulan di {CHART_DIR}")


if __name__ == "__main__":
    main()
