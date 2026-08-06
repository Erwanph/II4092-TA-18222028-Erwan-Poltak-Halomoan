"""Inti pipeline evaluasi kanonis (split 60/20/20, NN = ensembel 3-seed beku).

Modul ini menjadi sumber angka kanonis tesis: penyiapan timeline, pelatihan
model terbaik per algoritma, metrik uji, serta kontribusi SHAP/LIME per
observasi. Dipakai oleh generate_narratives (teks narasi LLM),
generate_xai_charts (gambar interpretasi), dan skrip analisis.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

from src.modeling.config import DATASET_PATH, TARGET_COLUMN
from src.modeling.data_prep import (
    load_dataset, handle_missing_values, create_sequences,
)
from src.modeling.metrics import (
    calculate_rmse, calculate_mape, r2_score, snap_to_grid,
    directional_metrics,
)
from src.modeling.machine_learning import RandomForestModel, XGBoostModel
from src.experiments.feature_engineering import apply_feature_engineering
from src.experiments.scenario_config import (
    generate_experiment_grid, build_experiment_config, DEFAULT_HYPERPARAMS,
    ALL_FEATURES,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

DATA_2026 = DATASET_PATH.parent / "dataset_2026.csv"
EVAL_DIR = Path("results/experiments/evaluasi")
SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
ENSEMBLE_SEEDS = [42, 43, 44]
STEP = 0.25
MODEL_ORDER = ["BiLSTM", "LSTM", "XGBoost", "RF"]
MONTHS = {1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
          7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November",
          12: "Desember"}
ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun", 7: "Jul",
        8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"}

# Konfigurasi model terbaik per algoritma (selaras registry & pred_vs_actual).
SPECS = {
    "BiLSTM": {"exp": "E136", "kind": "BiLSTM",
               "p": {"units": 128, "dropout": 0.4682, "sequence_length": 3,
                     "learning_rate": 0.001702, "batch_size": 16, "epochs": 100}},
    "LSTM": {"exp": "E144", "kind": "LSTM",
             "p": {"units": 32, "dropout": 0.4924, "sequence_length": 3,
                   "learning_rate": 0.009188, "batch_size": 8, "epochs": 100}},
    "RF": {"exp": "E136", "kind": "RF",
           "p": {"n_estimators": 50, "max_depth": None, "min_samples_split": 11,
                 "min_samples_leaf": 2, "max_features": 1.0, "random_state": 42}},
    "XGBoost": {"exp": "E112", "kind": "XGBoost",
                "p": dict(DEFAULT_HYPERPARAMS["XGBoost"], random_state=42)},
}

DISPLAY = {
    "BI_Rate_Pct_lag1": "BI-Rate t-1 (Persistensi)", "BI_Rate_Pct_lag2": "BI-Rate t-2",
    "BI_Rate_Pct_lag3": "BI-Rate t-3", "policy_regime": "Arah Kebijakan Suku Bunga",
    "real_rate": "Suku Bunga Riil", "output_gap": "Output Gap",
    "infl_momentum_3": "Momentum Inflasi (3 Bln)", "fx_pressure_3": "Tekanan Kurs (3 Bln)",
    "ffr_change": "Perubahan FFR", "fed_cycle": "Siklus The Fed",
    "Inflation_YoY_Pct": "Inflasi YoY", "Inflation_Gap_Pct": "Inflation Gap",
    "GDP_Growth_YoY_Pct": "Pertumbuhan PDB", "USD_IDR_Monthly_Avg": "Kurs USD/IDR",
    "M2_Triliun_Rp": "M2 (Uang Beredar)", "Credit_Growth_YoY_Pct": "Pertumbuhan Kredit",
    "Federal_Funds_Rate_Pct": "Federal Funds Rate", "IHSG_End_of_Month": "IHSG",
    "Foreign_Reserves_Miliar_USD": "Cadangan Devisa",
    "Oil_Price_Brent_USD_per_Bbl": "Harga Minyak Brent",
    "Gold_Price_USD_per_Oz": "Harga Emas", "VIX_Volatility_Index": "VIX",
}


def _disp(f):
    return DISPLAY.get(f, f)


def _f2(x):
    return ("n/a" if x is None else f"{x:.2f}").replace(".", ",")


def _f4(x):
    return ("n/a" if x is None else f"{x:.4f}").replace(".", ",")


def _grid():
    return {c["id"]: c for c in generate_experiment_grid()}


def prep(exp_id):
    """Rakit timeline kontinu (historis + holdout 2026) yang sudah terskala."""
    cfg = build_experiment_config(_grid()[exp_id])
    hist = load_dataset(DATASET_PATH)
    hold = pd.read_csv(DATA_2026)
    df = pd.concat([hist, hold], ignore_index=True)
    n_hold = len(hold)

    df = handle_missing_values(df, method=cfg["missing_method"],
                               knn_neighbors=cfg.get("knn_neighbors", 5))
    fe = cfg.get("feature_engineering")
    new = []
    if fe:
        df, new = apply_feature_engineering(df, fe, TARGET_COLUMN)
    feats = [c for c in list(cfg.get("feature_columns", ALL_FEATURES)) + new
             if c in df.columns]
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    n = len(df)
    n_hist = n - n_hold
    test_idx = int(n_hist * 0.8)          # 60/20/20 pada porsi historis
    val_idx = int(test_idx * 0.75)
    split = np.array(["train"] * val_idx + ["val"] * (test_idx - val_idx)
                     + ["test"] * (n_hist - test_idx) + ["holdout"] * n_hold)
    dates = [f"{int(t)}-{ABBR[int(b)]}" for b, t in zip(df["Bulan"], df["Tahun"])]

    Xraw = df[feats].values.astype(float)
    y = df[TARGET_COLUMN].values.astype(float)
    scaler = SCALERS[cfg["scaler"]]()
    scaler.fit(Xraw[:val_idx])            # fit hanya pada data latih
    X = scaler.transform(Xraw)
    return {"X": X, "Xraw": Xraw, "y": y, "dates": np.array(dates), "split": split,
            "feats": feats, "val_idx": val_idx, "test_idx": test_idx,
            "n_hist": n_hist, "scaler": scaler, "cfg": cfg}


class _DLEnsemble:
    """Ensembel NN 3-seed: prediksi = rata-rata anggota."""

    def __init__(self, members):
        self.members = members

    def predict(self, X, anchor=None):
        outs = [m.predict(X, anchor=anchor) for m in self.members]
        return np.mean(outs, axis=0)


def train_predict(name, d):
    """Latih model pada data latih, kembalikan prediksi (mentah+snap) seluruh timeline
    dan objek terlatih untuk SHAP."""
    X, y = d["X"], d["y"]
    vi = d["val_idx"]
    p = SPECS[name]["p"]
    kind = SPECS[name]["kind"]
    pred = np.full(len(y), np.nan)
    artefak = {}

    if kind in ("RF", "XGBoost"):
        # retrain fresh, deterministik (random_state=42); XGBoost + early stopping
        if kind == "RF":
            m = RandomForestModel(**p); m.fit(X[:vi], y[:vi])
        else:
            ti = d["test_idx"]
            m = XGBoostModel(**p); m.fit(X[:vi], y[:vi], X[vi:ti], y[vi:ti])
        pred = m.predict(X)
        artefak = {"kind": "tree", "model": m.model, "background": X[:vi]}
    else:
        from src.modeling.deep_learning import LSTMModel, BiLSTMModel
        seq = p["sequence_length"]
        Xseq, _ = create_sequences(X, y, seq)
        anc = y[seq - 1:-1]
        Xtr_s, ytr_s = create_sequences(X[:vi], y[:vi], seq)
        anc_tr = np.asarray(y[:vi], float)[seq - 1:-1]
        Cls = LSTMModel if kind == "LSTM" else BiLSTMModel
        exp_id = SPECS[name]["exp"]

        def _new():
            return Cls(sequence_length=seq, n_features=Xseq.shape[2], units=p["units"],
                       dropout=p["dropout"], learning_rate=p["learning_rate"],
                       batch_size=p["batch_size"], epochs=p["epochs"],
                       residual=True, seed=42)

        # persistensi lewat BOBOT (.weights.h5), bukan .keras — round-trip .keras
        # untuk Bidirectional LSTM rusak di Keras 3
        paths = [Path("models") / f"{exp_id}_tuned_{kind}_seed{s}.weights.h5"
                 for s in ENSEMBLE_SEEDS]
        members = []
        if all(pp.exists() for pp in paths):
            try:
                for pp in paths:
                    mm = _new(); mm.model.load_weights(str(pp))
                    members.append(mm)
            except Exception as e:
                print(f"  [NN {kind}] gagal memuat bobot seed ({e}); melatih ulang dari awal.")
                members = []
        if not members:
            for s, pp in zip(ENSEMBLE_SEEDS, paths):
                mm = Cls(sequence_length=seq, n_features=Xseq.shape[2], units=p["units"],
                         dropout=p["dropout"], learning_rate=p["learning_rate"],
                         batch_size=p["batch_size"], epochs=p["epochs"],
                         residual=True, seed=s)
                mm.fit(Xtr_s, ytr_s, anchor_train=anc_tr)
                pp.parent.mkdir(parents=True, exist_ok=True)
                mm.model.save_weights(str(pp))
                members.append(mm)
        last = _DLEnsemble(members)
        pred[seq:] = last.predict(Xseq, anchor=anc)
        artefak = {"kind": "nn", "model": last, "seq": seq, "Xseq": Xseq,
                   "anchor": anc, "background": Xtr_s}
    pred_snap = np.where(np.isnan(pred), np.nan, snap_to_grid(pred, STEP))
    return pred, pred_snap, artefak


def shap_instance(name, d, art, row):
    """Kontribusi SHAP pada satu observasi (indeks `row` di timeline).

    Tree: TreeExplainer (eksak). NN: KernelExplainer atas urutan yang diratakan,
    lalu diagregasi per fitur (jumlah lintas langkah waktu). Dikembalikan dict
    {fitur: kontribusi} terurut |kontribusi| menurun, atau None bila gagal.
    """
    feats = d["feats"]
    try:
        import shap
        if art["kind"] == "tree":
            is_xgb = art["model"].__class__.__name__ == "XGBRegressor"
            if is_xgb:
                # XGBoost 3.x breaks shap.TreeExplainer; use KernelExplainer via predict
                bg = art["background"]
                bg_s = bg[np.random.RandomState(0).choice(len(bg), min(50, len(bg)), replace=False)]
                expl_obj = shap.Explainer(art["model"].predict, bg_s)
                sv = expl_obj(d["X"][row].reshape(1, -1))
                vals = sv.values.ravel()
            else:
                expl = shap.TreeExplainer(art["model"])
                vals = np.asarray(expl.shap_values(d["X"][row].reshape(1, -1))).ravel()
            contrib = dict(zip(feats, vals[:len(feats)]))
        else:
            seq, Xseq = art["seq"], art["Xseq"]
            si = row - seq                      # indeks pada array sequence
            if si < 0:
                return None
            nf = len(feats)
            bg = art["background"]
            bg_s = bg[np.random.RandomState(0).choice(len(bg), min(30, len(bg)),
                                                       replace=False)]
            bg_flat = bg_s.reshape(bg_s.shape[0], -1)
            anchor_val = float(art["anchor"][si])

            def f(flat):
                arr = flat.reshape(flat.shape[0], seq, nf)
                return art["model"].predict(arr, anchor=np.full(len(arr), anchor_val))

            expl = shap.KernelExplainer(f, bg_flat)
            x_flat = Xseq[si].reshape(1, -1)
            sv = np.asarray(expl.shap_values(x_flat, nsamples=200)).reshape(seq, nf)
            contrib = dict(zip(feats, sv.sum(axis=0)))   # agregasi lintas waktu
        return dict(sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True))
    except Exception as e:
        print(f"  [SHAP {name}] dilewati: {e}")
        return None


def shap_explanation(name, d, art, row):
    """Seperti shap_instance tetapi mengembalikan (contrib_dict, Explanation)
    agar SHAP mahal (KernelExplainer NN) cukup dihitung sekali."""
    feats = d["feats"]
    disp = [_disp(f) for f in feats]
    try:
        import shap
        if art["kind"] == "tree":
            is_xgb = art["model"].__class__.__name__ == "XGBRegressor"
            x = d["X"][row].reshape(1, -1)
            if is_xgb:
                bg = art["background"]
                bg_s = bg[np.random.RandomState(0).choice(len(bg), min(50, len(bg)),
                                                          replace=False)]
                ex = shap.Explainer(art["model"].predict, bg_s)
                sv = ex(x)
            else:
                ex = shap.TreeExplainer(art["model"])
                sv = ex(x)
            vals = np.asarray(sv.values).ravel()[:len(feats)]
            base = float(np.ravel(sv.base_values)[0])
            data = np.asarray(sv.data).ravel()[:len(feats)]
        else:
            seq, Xseq = art["seq"], art["Xseq"]
            si = row - seq                      # indeks pada array sequence
            if si < 0:
                return None, None
            nf = len(feats)
            bg = art["background"]
            bg_s = bg[np.random.RandomState(0).choice(len(bg), min(30, len(bg)),
                                                       replace=False)]
            bg_flat = bg_s.reshape(bg_s.shape[0], -1)
            anchor_val = float(art["anchor"][si])

            def f(flat):
                arr = flat.reshape(flat.shape[0], seq, nf)
                return art["model"].predict(arr, anchor=np.full(len(arr), anchor_val))

            expl = shap.KernelExplainer(f, bg_flat)
            x_flat = Xseq[si].reshape(1, -1)
            sv = np.asarray(expl.shap_values(x_flat, nsamples=200)).reshape(seq, nf)
            vals = sv.sum(axis=0)                # agregasi kontribusi lintas waktu
            base = float(np.ravel(expl.expected_value)[0])
            data = Xseq[si].mean(axis=0)         # rata-rata nilai fitur lintas waktu
        contrib = dict(sorted(zip(feats, vals), key=lambda kv: abs(kv[1]), reverse=True))
        expl_obj = shap.Explanation(values=np.asarray(vals, dtype=float),
                                    base_values=base,
                                    data=np.asarray(data, dtype=float),
                                    feature_names=disp)
        return contrib, expl_obj
    except Exception as e:
        print(f"  [SHAP {name}] dilewati: {e}")
        return None, None


def lime_instance(name, d, art, row, nn_samples=1000):
    """Kontribusi LIME satu observasi (NN: urutan diratakan lalu diagregasi
    per fitur); pembanding silang untuk SHAP."""
    feats = d["feats"]
    try:
        import lime.lime_tabular
        if art["kind"] == "tree":
            bg = art["background"]
            expl = lime.lime_tabular.LimeTabularExplainer(
                bg, feature_names=feats, mode="regression", verbose=False,
                random_state=0)
            exp = expl.explain_instance(d["X"][row], art["model"].predict,
                                        num_features=len(feats))
            contrib = {feats[i]: float(w) for i, w in exp.local_exp[1]}
        else:
            seq, Xseq = art["seq"], art["Xseq"]
            si = row - seq                      # indeks pada array sequence
            if si < 0:
                return None
            nf = len(feats)
            bg = art["background"]
            bg_flat = bg.reshape(bg.shape[0], -1)
            anchor_val = float(art["anchor"][si])

            def f(flat):
                arr = flat.reshape(flat.shape[0], seq, nf)
                return art["model"].predict(arr, anchor=np.full(len(arr), anchor_val))

            flat_names = [f"{ft}@t{t}" for t in range(seq) for ft in feats]
            expl = lime.lime_tabular.LimeTabularExplainer(
                bg_flat, feature_names=flat_names, mode="regression",
                verbose=False, random_state=0)
            exp = expl.explain_instance(Xseq[si].reshape(-1), f,
                                        num_features=seq * nf, num_samples=nn_samples)
            agg = {ft: 0.0 for ft in feats}
            for idx, w in exp.local_exp[1]:        # agregasi lintas waktu
                agg[feats[idx % nf]] += float(w)
            contrib = agg
        return dict(sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True))
    except Exception as e:
        print(f"  [LIME {name}] dilewati: {e}")
        return None


def metrics_test(d, pred_snap, pred_raw):
    """Metrik pada himpunan uji + hit-rate arah (bulan-perubahan)."""
    m = d["split"] == "test"
    y = d["y"][m]
    ps = pred_snap[m]
    pr = pred_raw[m]
    ok = ~np.isnan(ps)
    y, ps, pr = y[ok], ps[ok], pr[ok]
    rmse = calculate_rmse(y, pr)
    mape = calculate_mape(y, pr)
    r2 = r2_score(y, pr)
    y_prev = d["y"][np.where(m)[0] - 1]
    dm = directional_metrics(y, ps, y_prev)
    # hit-rate khusus bulan-perubahan (arah benar di antara bulan yang berubah)
    dir_true = np.sign(np.round((y - y_prev) / STEP))
    dir_pred = np.sign(np.round((ps - y_prev) / STEP))
    chg = dir_true != 0
    hit_change = float(np.mean(dir_true[chg] == dir_pred[chg])) if chg.any() else None
    return {"RMSE": rmse, "MAPE": mape, "R2": r2,
            "hit_all": dm["direction_hit_rate"], "hit_change": hit_change,
            "n_change": int(chg.sum())}


def pick_control_months(trained):
    """Pilih dua bulan kontrol yang sama untuk semua model: satu bulan-perubahan
    yang diprediksi tepat oleh semua model dan satu yang meleset di semua model.
    """
    ref = next(iter(trained.values()))["d"]
    y, dates, split = ref["y"], ref["dates"], ref["split"]
    names = list(trained)
    cands = []
    for i in range(1, len(y)):
        if split[i] != "test":
            continue
        if any(np.isnan(trained[n]["pred_snap"][i]) for n in names):
            continue
        prev, act = float(y[i - 1]), float(y[i])
        if abs(act - prev) <= 1e-9:        # hanya bulan-perubahan kebijakan
            continue
        errs = [float(trained[n]["pred_snap"][i]) - act for n in names]
        aerr = [abs(e) for e in errs]
        miss = [e for e in errs if abs(e) > 1e-9]
        same_dir = bool(miss) and (all(e < 0 for e in miss) or all(e > 0 for e in miss))
        cands.append({"i": i, "label": dates[i], "prev": prev, "actual": act,
                      "n_correct": sum(1 for e in aerr if e < 1e-9),
                      "all_correct": all(e < 1e-9 for e in aerr),
                      "all_miss": all(e > 1e-9 for e in aerr),
                      "spread": max(aerr) - min(aerr), "same_dir": same_dir})
    if not cands:
        return []

    allc = [c for c in cands if c["all_correct"]]
    tepat = (max(allc, key=lambda c: c["i"]) if allc
             else max(cands, key=lambda c: (c["n_correct"], c["i"])))

    allm = [c for c in cands if c["all_miss"] and c["i"] != tepat["i"]]
    if allm:
        meleset = min(allm, key=lambda c: (0 if c["same_dir"] else 1, c["spread"], -c["i"]))
    else:
        rest = [c for c in cands if c["i"] != tepat["i"]]
        meleset = (min(rest, key=lambda c: (c["n_correct"], -c["i"])) if rest else None)

    out = [dict(tepat, kategori="tepat")]
    if meleset:
        out.append(dict(meleset, kategori="meleset"))
    return out


# ----------------------------- Narasi LLM -----------------------------
def _get_llm():
    try:
        from src.xai.llm_narrator import LLMNarrativeGenerator
        return LLMNarrativeGenerator()
    except Exception as e:
        print(f"Narasi LLM dilewati ({e}); memakai ringkasan deterministik.")
        return None


def _feature_facts(d, row, contrib, topn=6):
    """Rakit fakta kuantitatif (nilai aktual, perubahan bulan-ke-bulan, status ekstrem)
    untuk kontributor SHAP teratas, dari matriks fitur MENTAH `Xraw`. Memberi LLM
    bahan konkret ("dari X ke Y", "tertinggi sejak 2005") tanpa keluar dari data."""
    from src.xai.llm_narrator import build_feature_fact
    feats, Xraw = d["feats"], d["Xraw"]
    facts = {}
    for raw, _ in list((contrib or {}).items())[:topn]:
        if raw not in feats:
            continue
        j = feats.index(raw)
        val = float(Xraw[row, j])
        prev = float(Xraw[row - 1, j]) if row > 0 else None
        facts[raw] = build_feature_fact(raw, val, prev, Xraw[:, j])
    return facts


def _period_id(label):
    """'2025-Jul' -> 'Juli 2025' untuk narasi yang lebih terbaca."""
    try:
        thn, abb = str(label).split("-")
        num = {v: k for k, v in ABBR.items()}.get(abb)
        return f"{MONTHS.get(num, abb)} {thn}"
    except Exception:
        return str(label)


def _deterministic_narrative(pred, contrib, facts, period, prev=None):
    """Narasi naratif yang mengalir dan mudah dipahami pakar ekonomi (tanpa istilah
    teknis machine learning). Kekuatan tiap faktor dinyatakan sebagai PERSENTASE dari
    total pengaruh bulan itu, dilengkapi nilai aktual dan perubahannya (dari X menjadi
    Y). Sepenuhnya grounded pada angka yang tersedia, tanpa menambah informasi luar."""
    from src.xai.llm_narrator import describe_feature, _fmt_id
    per = _period_id(period) if period else "bulan ini"
    pred_s = f"{pred:.2f}".replace(".", ",")
    if not contrib:
        return (f"Pada {per}, model memperkirakan BI-Rate berada di sekitar {pred_s}%. "
                "Rincian faktor pendorong tidak tersedia untuk bulan ini.")
    items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
    total = sum(abs(v) for _, v in items) or 1.0

    def persen(v):
        return max(1, round(abs(v) / total * 100))

    def klausa(raw, v, sebut_def=True):
        """Satu klausa naratif untuk sebuah faktor: nama awam (+definisi), nilai aktual
        dan perubahannya, lalu besar pengaruhnya dalam persen."""
        name, defn = describe_feature(raw)
        f = (facts or {}).get(raw, {})
        s = name
        if sebut_def and defn:
            s += f", yaitu {defn},"
        if f.get("value") is not None:
            nilai = _fmt_id(f["value"], f.get("unit", ""))
            if f.get("delta") is not None and abs(f["delta"]) > 1e-9:
                ad = "naik" if f["delta"] > 0 else "turun"
                s += (f" yang pada bulan ini {ad} dari "
                      f"{_fmt_id(f.get('prev'), f.get('unit', ''))} menjadi {nilai}")
            else:
                s += f" yang pada bulan ini tercatat {nilai}"
            if f.get("extreme"):
                s += f" dan {f['extreme']}"
        s += f", dengan pengaruh sekitar {persen(v)}% terhadap prediksi bulan ini"
        return s

    pos = [(k, v) for k, v in items if v > 0]
    neg = [(k, v) for k, v in items if v < 0]
    dom_key, dom_val = items[0]
    dom_name = describe_feature(dom_key)[0]

    # Deteksi faktor persistensi (BI-Rate bulan sebelumnya) & level acuannya
    def _is_persist(k):
        return "BI_Rate" in k and "lag1" in k
    persist = next((kv for kv in items if _is_persist(kv[0])), None)
    prev_level = None
    if persist is not None:
        prev_level = (facts or {}).get(persist[0], {}).get("value")
    if prev_level is None and prev is not None:
        # Fallback: level bulan sebelumnya dari data (mis. model tanpa fitur lag)
        prev_level = prev

    # 1) Pembingkaian sebagai keputusan kebijakan terhadap bulan sebelumnya
    s = f"Pada {per}, model memperkirakan BI-Rate berada di sekitar {pred_s}%. "
    if prev_level is not None:
        diff = pred - prev_level
        prev_s = f"{prev_level:.2f}".replace(".", ",")
        if abs(diff) < 0.125:
            arah = (f"Dibandingkan posisi bulan sebelumnya yang berada di {prev_s}%, "
                    "angka ini mengisyaratkan kecenderungan untuk MEMPERTAHANKAN suku bunga "
                    "pada level yang sama")
        elif diff > 0:
            arah = (f"Dibandingkan posisi bulan sebelumnya yang berada di {prev_s}%, "
                    "angka ini mengisyaratkan kecenderungan untuk MENAIKKAN suku bunga")
        else:
            arah = (f"Dibandingkan posisi bulan sebelumnya yang berada di {prev_s}%, "
                    "angka ini mengisyaratkan kecenderungan untuk MENURUNKAN suku bunga")
        s += arah + ". "
    s += ("Prediksi ini bukan tebakan tunggal, melainkan hasil tarik-menarik antara faktor "
          "yang mendorong suku bunga naik dan faktor yang menahannya tetap rendah. Bagian "
          "berikut menguraikan faktor paling berperan, seberapa besar bobotnya, serta arah "
          "pengaruhnya, agar alasan di balik angka prediksi dapat ditelusuri.\n\n")

    # 2) Mengapa persistensi mendominasi (bila benar dominan)
    if persist is not None and _is_persist(dom_key):
        s += (f"Penentu utama prediksi adalah level BI-Rate bulan sebelumnya, dengan bobot "
              f"sekitar {persen(dom_val)}% dari total pengaruh. Hal ini mencerminkan watak "
              "khas kebijakan suku bunga yang bergerak gradual dan jarang berubah mendadak "
              "(penghalusan suku bunga / interest-rate smoothing): titik awal "
              "yang paling masuk akal bagi keputusan bulan ini adalah posisi bulan lalu, lalu "
              "indikator makroekonomi memberi koreksi kecil di sekitarnya.\n\n")

    # 3) Pendorong ke arah lebih tinggi
    if pos:
        s += ("Dorongan ke arah lebih tinggi paling kuat berasal dari "
              + klausa(*pos[0], sebut_def=True) + ".")
        if len(pos) > 1:
            s += (" Selain itu, ikut mendorong naik "
                  + "; ".join(klausa(k, v, sebut_def=False) for k, v in pos[1:4]) + ".")
        s += "\n\n"

    # 4) Penahan ke arah lebih rendah
    if neg:
        s += ("Di sisi lain, faktor yang menahan prediksi agar tidak lebih tinggi terutama "
              "datang dari " + klausa(*neg[0], sebut_def=True) + ".")
        if len(neg) > 1:
            s += (" Tekanan ke bawah yang serupa juga datang dari "
                  + "; ".join(klausa(k, v, sebut_def=False) for k, v in neg[1:4]) + ".")
        s += "\n\n"

    # 5) Neraca bersih: kaitkan keseimbangan dorongan dgn arah prediksi
    net = sum(v for _, v in items)
    if pos and neg:
        if net > 1e-6:
            kesimpulan_neraca = ("dorongan ke atas sedikit lebih kuat daripada penahannya, "
                                 "sehingga prediksi condong ke arah lebih tinggi")
        elif net < -1e-6:
            kesimpulan_neraca = ("penahan ke bawah sedikit lebih kuat daripada pendorongnya, "
                                 "sehingga prediksi condong ke arah lebih rendah")
        else:
            kesimpulan_neraca = ("kedua sisi nyaris seimbang, sehingga prediksi cenderung "
                                 "bertahan di sekitar level sebelumnya")
        s += (f"Bila ditimbang, total dorongan ke atas dan ke bawah hampir berimbang "
              f"({kesimpulan_neraca}). ")

    # 6) Benang merah
    s += (f"Secara keseluruhan, prediksi BI-Rate {per} paling ditentukan oleh "
          f"{dom_name} (pengaruh terbesar, sekitar {persen(dom_val)}%), sementara "
          "faktor-faktor makroekonomi lain bekerja saling mengimbangi di sekitarnya untuk "
          "memberi koreksi halus.\n\n")

    # 7) Catatan
    s += ("Perlu dicatat bahwa penjelasan ini disusun langsung dari pola yang dipelajari "
          "model atas data historis dan nilai indikator aktual bulan tersebut, tanpa "
          "menambahkan informasi dari luar data; interpretasi ekonomi yang lebih mendalam "
          "tetap memerlukan pertimbangan pakar terkait.")
    return s


def _narrative(llm, pred, contrib, facts=None, period=None, prev=None):
    """Narasi interpretasi 1 titik; fallback deterministik bila LLM mati.
    `contrib` memakai nama fitur mentah agar glosarium bisa menerjemahkannya."""
    if llm is not None:
        try:
            return llm.generate_narrative(pred, contrib, feature_facts=facts, period=period)
        except Exception as e:
            print(f"  Narasi LLM gagal ({e}); memakai ringkasan deterministik.")
    return _deterministic_narrative(pred, contrib, facts, period, prev=prev)
