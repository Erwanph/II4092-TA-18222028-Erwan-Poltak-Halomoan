"""Narasi LLM (teks) untuk seluruh bulan evaluasi, satu perintah.

Melatih keempat model kanonis, memilih dua bulan kontrol yang sama (satu bulan
seluruh model tepat, satu bulan seluruh model meleset), menghitung kontribusi
SHAP tiap model pada bulan itu, lalu menulis narasi llm_narrator sebagai berkas
teks di results/experiments/evaluasi/narasi/. Teks inilah yang dikutip Lampiran
dan disalin penulis ke lembar evaluasi domain expert.
BERAT -- jalankan manual: python -m src.generate_narratives
"""

import numpy as np

from src.evaluation_core import (
    SPECS, MODEL_ORDER, EVAL_DIR, STEP,
    prep, train_predict, metrics_test, pick_control_months,
    shap_instance, _feature_facts, _narrative, _get_llm, _f2, _f4,
)

NARASI_DIR = EVAL_DIR / "narasi"


def _drivers_line(contrib, topn=5):
    """Ringkasan faktor teratas (arah + persen bobot) untuk kepala berkas teks."""
    if not contrib:
        return "-"
    from src.xai.llm_narrator import describe_feature
    items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
    tot = sum(abs(v) for _, v in items) or 1.0
    parts = []
    for k, v in items[:topn]:
        arah = "mendorong naik" if v > 0 else "menahan turun"
        p = max(1, round(abs(v) / tot * 100))
        parts.append(f"{describe_feature(k)[0]} ({arah}, ~{p}%)")
    return "; ".join(parts)

    # d0 = trained[MODEL_ORDER[0]]["d"]
    # i = int(np.where(d0["dates"] == "2026-Mei")[0][0])
    # controls = [{"i": i, "label": str(d0["dates"][i]), "kategori": "khusus", "prev": float(d0["y"][i - 1]), "actual": float(d0["y"][i])}]

def main():
    NARASI_DIR.mkdir(parents=True, exist_ok=True)
    llm = _get_llm()

    # 1) Latih seluruh model dahulu agar bulan kontrol dapat dipilih lintas-model.
    trained = {}
    for name in MODEL_ORDER:
        print(f"\n=== Latih {name} ===")
        d = prep(SPECS[name]["exp"])
        pred_raw, pred_snap, art = train_predict(name, d)
        met = metrics_test(d, pred_snap, pred_raw)
        print(f"  Uji: RMSE={_f4(met['RMSE'])} MAPE={_f2(met['MAPE'])}% R2={_f4(met['R2'])}")
        trained[name] = {"d": d, "pred_snap": pred_snap, "art": art, "met": met}

    # 2) Pilih bulanm
    controls = pick_control_months(trained)
    for c in controls:
        print(f"Bulan kontrol [{c['kategori']}]: {c['label']} (aktual {c['actual']})")

    # 3) Narasi tiap model pada kedua bulan kontrol -> berkas teks.
    for name in MODEL_ORDER:
        t = trained[name]
        for c in controls:
            i = c["i"]
            pred = float(t["pred_snap"][i])
            contrib = shap_instance(name, t["d"], t["art"], i)
            facts = _feature_facts(t["d"], i, contrib)
            narrative = _narrative(llm, pred, contrib or {}, facts, c["label"],
                                   prev=c["prev"])
            dir_ok = (np.sign(round((pred - c["prev"]) / STEP))
                      == np.sign(round((c["actual"] - c["prev"]) / STEP)))
            head = (
                f"Model     : {name}\n"
                f"Bulan     : {c['label']} (kontrol {c['kategori']})\n"
                f"Aktual    : {_f2(c['actual'])}% (bulan sebelumnya {_f2(c['prev'])}%)\n"
                f"Prediksi  : {_f2(pred)}% "
                f"({'arah tepat' if dir_ok else 'arah meleset'})\n"
                f"Faktor    : {_drivers_line(contrib)}\n"
                + "-" * 72 + "\n\n"
            )
            out = NARASI_DIR / f"{name}_{c['label']}.txt"
            out.write_text(head + narrative + "\n", encoding="utf-8")
            print(f"  -> {out}")
    print(f"\nSelesai. Narasi teks per model/bulan di {NARASI_DIR}")


if __name__ == "__main__":
    main()
