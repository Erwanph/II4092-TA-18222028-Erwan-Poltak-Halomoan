"""
PDF Report Generator for BI-Rate factorial experiment results.
Reads screening, focused, tuning, and final CSV outputs to produce
a comprehensive, narrative-driven experiment report.
"""

import json
import pandas as pd
from pathlib import Path

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


def sanitize_text(text):
    """Replace Unicode characters that Helvetica cannot render."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2022": "-", "\u2026": "...",
        "\u00d7": "x", "\u2264": "<=", "\u2265": ">=",
        "\u2192": "->", "\u2605": "*",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="replace").decode("latin-1")


# Human-readable labels for factor levels
LABEL_MAP = {
    "interpolate": "Interpolasi Linear",
    "ffill": "Forward Fill",
    "mean": "Mean Imputation",
    "knn": "KNN Imputation",
    "all": "Semua 12 Fitur",
    "domain_itf": "Domain ITF (7 Fitur)",
    "corr_top6": "Top-6 Korelasi",
    "mutual_info": "Mutual Info Top-7",
    "standard": "StandardScaler",
    "minmax": "MinMaxScaler",
    "robust": "RobustScaler",
    "none": "Tanpa FE",
    "lag": "Lag (t-1, t-3, t-6)",
    "rolling": "Rolling Mean/Std",
    "lag_rolling": "Lag + Rolling",
}


def _label(key):
    return LABEL_MAP.get(key, str(key))


class ExperimentReportPDF(FPDF):
    """Custom PDF class for experiment reports."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "Laporan Hasil Eksperimen Prediksi BI-Rate",
                  align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(41, 128, 185)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(41, 128, 185)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, f"Halaman {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(41, 128, 185)
        self.cell(0, 10, sanitize_text(title),
                  new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(41, 128, 185)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 90, self.get_y())
        self.ln(3)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(44, 62, 80)
        self.cell(0, 8, sanitize_text(title),
                  new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, sanitize_text(text))
        self.ln(3)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 7)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, sanitize_text(str(h)),
                      border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 7)
        self.set_text_color(50, 50, 50)
        for idx, row in enumerate(rows):
            bg = (240, 248, 255) if idx % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*bg)
            for i, val in enumerate(row):
                align = "C" if i > 0 else "L"
                self.cell(col_widths[i], 6, sanitize_text(str(val)),
                          border=1, fill=True, align=align)
            self.ln()
        self.ln(4)

    def combo_detail_block(self, row):
        """Print a compact summary of one experiment combination."""
        self.set_font("Helvetica", "", 9)
        self.set_text_color(60, 60, 60)
        imp = _label(row.get("imputation", "?"))
        feat = _label(row.get("features", "?"))
        sc = _label(row.get("scaler", "?"))
        fe = _label(row.get("feature_engineering", "?"))
        text = (f"Imputasi: {imp}  |  Fitur: {feat}  "
                f"|  Scaler: {sc}  |  Feature Engineering: {fe}")
        self.cell(0, 5, sanitize_text(text),
                  new_x="LMARGIN", new_y="NEXT")
        self.ln(1)


def _write_cover(pdf, n_screening, n_focused, n_models):
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 14, "Laporan Eksperimen", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 14, "Prediksi BI-Rate", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Desain Eksperimen Faktorial (CRISP-DM)",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    info_lines = [
        f"Total Kombinasi Screening: {n_screening}",
        f"Kombinasi Terfokus (Focused): {n_focused}",
        f"Jumlah Model Diuji: {n_models}",
    ]
    for line in info_lines:
        pdf.cell(0, 7, sanitize_text(line), align="C",
                 new_x="LMARGIN", new_y="NEXT")

    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 6, "Framework Eksperimen Terstruktur berbasis CRISP-DM",
             align="C", new_x="LMARGIN", new_y="NEXT")


def generate_experiment_report(results_dir, output_path=None):
    """Generate comprehensive experiment report PDF from pipeline outputs."""
    if not FPDF_AVAILABLE:
        print("fpdf2 diperlukan. Install: pip install fpdf2")
        return None

    results_dir = Path(results_dir)
    output_path = output_path or results_dir / "experiment_report_final.pdf"

    screening_path = results_dir / "screening" / "screening_results.csv"
    grid_path = results_dir / "screening" / "experiment_grid.json"
    analysis_path = results_dir / "analysis" / "factor_analysis.json"
    focused_path = results_dir / "focused" / "focused_results.csv"
    tuning_path = results_dir / "tuning" / "tuning_results.csv"
    final_path = results_dir / "final" / "final_comparison.csv"

    screening_df = pd.read_csv(screening_path) if screening_path.exists() else pd.DataFrame()
    focused_df = pd.read_csv(focused_path) if focused_path.exists() else pd.DataFrame()
    tuning_df = pd.read_csv(tuning_path) if tuning_path.exists() else pd.DataFrame()
    final_df = pd.read_csv(final_path) if final_path.exists() else pd.DataFrame()

    grid_dict = {}
    if grid_path.exists():
        with open(grid_path, encoding="utf-8") as f:
            grid_dict = {c["id"]: c for c in json.load(f)}

    factor_analysis = {}
    if analysis_path.exists():
        with open(analysis_path, encoding="utf-8") as f:
            factor_analysis = json.load(f)

    n_screening = screening_df["experiment_id"].nunique() if not screening_df.empty else 0
    n_focused = focused_df["experiment_id"].nunique() if not focused_df.empty else 0
    all_models = list(focused_df["model"].unique()) if not focused_df.empty else []

    pdf = ExperimentReportPDF()
    pdf.alias_nb_pages()

    _write_cover(pdf, n_screening, n_focused, len(all_models))

    # Ringkasan Eksekutif
    pdf.add_page()
    pdf.section_title("Ringkasan Eksekutif")
    pdf.body_text(
        "Laporan ini menyajikan hasil lengkap dari pipeline eksperimen faktorial "
        "untuk memprediksi BI-Rate (suku bunga acuan Bank Indonesia). Pendekatan "
        "yang digunakan adalah desain faktorial penuh, di mana setiap kombinasi "
        "dari empat faktor utama (metode imputasi, seleksi fitur, jenis scaler, "
        "dan rekayasa fitur) diuji secara sistematis."
    )
    pdf.body_text(
        "Eksperimen terdiri dari lima tahap berurutan: "
        "(1) Screening awal dengan model cepat untuk menyaring 192 kombinasi, "
        "(2) Analisis faktor untuk memahami kontribusi tiap komponen, "
        "(3) Evaluasi mendalam pada kombinasi terbaik dengan seluruh model "
        "(termasuk ARIMA, VAR, Random Forest, XGBoost, LSTM, dan BiLSTM), "
        "(4) Optimasi hyperparameter pada kandidat unggulan, dan "
        "(5) Perbandingan akhir antara konfigurasi optimal dan baseline."
    )

    # Section 1: Screening
    if not screening_df.empty:
        pdf.add_page()
        pdf.section_title("1. Hasil Screening (Fase 1)")
        pdf.body_text(
            f"Pada tahap screening, sebanyak {n_screening} kombinasi eksperimen "
            f"dievaluasi menggunakan model RF dan XGBoost. Tabel berikut menampilkan "
            f"20 kombinasi dengan rata-rata RMSE terendah, beserta detail konfigurasi "
            f"masing-masing skenario."
        )

        avg_per_exp = (screening_df.groupby("experiment_id")["RMSE"]
                       .mean().sort_values().head(20))

        headers = ["No", "ID", "Imputasi", "Fitur", "Scaler", "FE", "RMSE"]
        rows = []
        for rank, (exp_id, rmse) in enumerate(avg_per_exp.items(), 1):
            combo = grid_dict.get(exp_id, {})
            rows.append([
                str(rank),
                exp_id,
                _label(combo.get("imputation", "?")),
                _label(combo.get("features", "?")),
                _label(combo.get("scaler", "?")),
                _label(combo.get("feature_engineering", "?")),
                f"{rmse:.4f}",
            ])
        pdf.add_table(headers, rows,
                      col_widths=[10, 16, 32, 32, 30, 36, 24])

        pdf.body_text(
            f"Dari hasil screening di atas, kombinasi {avg_per_exp.index[0]} "
            f"mencatat rata-rata RMSE paling rendah ({avg_per_exp.iloc[0]:.4f}), "
            f"mengindikasikan bahwa konfigurasi tersebut memberikan prediksi "
            f"yang paling konsisten di antara dua model screening."
        )

    # Section 2: Analisis Faktor
    if factor_analysis:
        pdf.add_page()
        pdf.section_title("2. Analisis Pengaruh Faktor (Fase 2)")
        pdf.body_text(
            "Analisis ini mengukur kontribusi marginal dari setiap faktor "
            "terhadap performa prediksi. Semakin besar rentang rata-rata RMSE "
            "antar level suatu faktor, semakin signifikan pengaruhnya."
        )

        importance = factor_analysis.get("importance", {})
        if importance:
            imp_sorted = sorted(importance.items(),
                                key=lambda x: x[1]["range"], reverse=True)
            headers = ["No", "Faktor", "Rentang RMSE",
                       "Level Terbaik", "Level Terburuk"]
            rows = []
            for rank, (f, info) in enumerate(imp_sorted, 1):
                rows.append([
                    str(rank), f.replace("_", " ").title(),
                    f"{info['range']:.4f}",
                    _label(info["best_level"]),
                    _label(info["worst_level"]),
                ])
            pdf.add_table(headers, rows,
                          col_widths=[12, 42, 30, 50, 50])

            most_imp = imp_sorted[0]
            least_imp = imp_sorted[-1]
            pdf.body_text(
                f"Faktor yang paling berpengaruh adalah "
                f"'{most_imp[0].replace('_', ' ')}' dengan rentang RMSE "
                f"sebesar {most_imp[1]['range']:.4f}, di mana level "
                f"'{_label(most_imp[1]['best_level'])}' mengungguli "
                f"'{_label(most_imp[1]['worst_level'])}'. "
                f"Sementara itu, faktor '{least_imp[0].replace('_', ' ')}' "
                f"memiliki pengaruh paling kecil "
                f"(rentang {least_imp[1]['range']:.4f}), yang berarti "
                f"pemilihan level pada faktor tersebut relatif tidak "
                f"mengubah performa secara drastis."
            )

    # Section 3: Evaluasi Mendalam (Focused)
    if not focused_df.empty:
        pdf.add_page()
        pdf.section_title("3. Evaluasi Mendalam (Fase 3)")

        n_combos = focused_df["experiment_id"].nunique()
        models_used = ", ".join(focused_df["model"].unique())
        pdf.body_text(
            f"Pada fase ini, {n_combos} kombinasi terbaik dari hasil screening "
            f"dievaluasi menggunakan seluruh model yang tersedia: {models_used}. "
            f"Tabel berikut menampilkan 15 hasil terbaik secara keseluruhan."
        )

        top_15 = focused_df.nsmallest(15, "RMSE")
        headers = ["No", "ID", "Model", "Imputasi", "Fitur",
                    "Scaler", "FE", "RMSE", "MAPE", "R2"]
        rows = []
        for rank, (_, r) in enumerate(top_15.iterrows(), 1):
            rows.append([
                str(rank),
                r["experiment_id"],
                r["model"],
                _label(r.get("imputation", "?")),
                _label(r.get("features", "?"))[:14],
                _label(r.get("scaler", "?")),
                _label(r.get("feature_engineering", "?"))[:14],
                f"{r['RMSE']:.4f}",
                f"{r['MAPE']:.2f}%",
                f"{r['R2']:.4f}",
            ])
        pdf.add_table(headers, rows,
                      col_widths=[8, 14, 20, 24, 24, 24, 24, 16, 18, 16])

        best_row = focused_df.loc[focused_df["RMSE"].idxmin()]
        pdf.sub_title("Detail Konfigurasi Terbaik")
        pdf.combo_detail_block(best_row)
        pdf.body_text(
            f"Model {best_row['model']} dengan konfigurasi {best_row['experiment_id']} "
            f"menghasilkan RMSE terendah sebesar {best_row['RMSE']:.4f}, "
            f"MAPE sebesar {best_row['MAPE']:.2f}%, dan R2 sebesar {best_row['R2']:.4f}. "
            f"Konfigurasi ini menjadi kandidat utama untuk tahap optimasi selanjutnya."
        )

        # Per-model best
        pdf.sub_title("Performa Terbaik per Model")
        best_per_model = focused_df.loc[
            focused_df.groupby("model")["RMSE"].idxmin()
        ].sort_values("RMSE")
        headers = ["Model", "ID Kombinasi", "Imputasi", "Fitur",
                    "Scaler", "FE", "RMSE", "R2"]
        rows = []
        for _, r in best_per_model.iterrows():
            rows.append([
                r["model"],
                r["experiment_id"],
                _label(r.get("imputation", "?")),
                _label(r.get("features", "?"))[:14],
                _label(r.get("scaler", "?")),
                _label(r.get("feature_engineering", "?"))[:14],
                f"{r['RMSE']:.4f}",
                f"{r['R2']:.4f}",
            ])
        pdf.add_table(headers, rows,
                      col_widths=[20, 18, 28, 24, 24, 24, 24, 24])

    # Section 4: Hyperparameter Tuning
    if not tuning_df.empty:
        pdf.add_page()
        pdf.section_title("4. Optimasi Hyperparameter (Fase 4)")
        pdf.body_text(
            "Setelah kombinasi data processing terbaik terpilih, model-model "
            "unggulan menjalani proses optimasi hyperparameter menggunakan "
            "Bayesian Optimization (Optuna). Tabel berikut membandingkan "
            "performa sebelum dan sesudah tuning."
        )

        headers = ["ID", "Model", "Kombinasi Percobaan",
                    "RMSE Awal", "RMSE Tuned", "Perubahan"]
        rows = []
        for _, r in tuning_df.iterrows():
            default_rmse = r.get("RMSE_default")
            tuned_rmse = r.get("RMSE_tuned")
            improv = "-"
            if pd.notnull(default_rmse) and pd.notnull(tuned_rmse) and default_rmse > 0:
                pct = ((default_rmse - tuned_rmse) / default_rmse) * 100
                improv = f"{pct:+.1f}%"

            combo = grid_dict.get(r["experiment_id"], {})
            combo_str = (f"{_label(combo.get('imputation','?'))} + "
                         f"{_label(combo.get('features','?'))}")

            rows.append([
                r["experiment_id"],
                r["model"],
                combo_str[:30],
                f"{default_rmse:.4f}" if pd.notnull(default_rmse) else "-",
                f"{tuned_rmse:.4f}" if pd.notnull(tuned_rmse) else "-",
                improv,
            ])
        pdf.add_table(headers, rows,
                      col_widths=[16, 22, 60, 26, 26, 26])

        improved = tuning_df.dropna(subset=["RMSE_default", "RMSE_tuned"])
        if not improved.empty:
            improved = improved[improved["RMSE_tuned"] < improved["RMSE_default"]]
            if not improved.empty:
                avg_improvement = (
                    (improved["RMSE_default"] - improved["RMSE_tuned"])
                    / improved["RMSE_default"] * 100
                ).mean()
                pdf.body_text(
                    f"Dari {len(improved)} konfigurasi yang berhasil ditingkatkan, "
                    f"rata-rata penurunan RMSE setelah tuning adalah "
                    f"sebesar {avg_improvement:.1f}%."
                )

    # Section 5: Perbandingan Final
    if not final_df.empty:
        pdf.add_page()
        pdf.section_title("5. Perbandingan Final (Fase 5)")
        pdf.body_text(
            "Tahap akhir ini membandingkan seluruh model menggunakan tiga "
            "konfigurasi: (1) Baseline dengan pengaturan default "
            "(interpolasi linear, semua fitur, StandardScaler, tanpa FE), "
            "(2) Konfigurasi optimal yang dijalankan pada sesi eksperimen saat ini, dan "
            "(3) Hasil terbaik sepanjang seluruh eksperimen yang tercatat dalam "
            "registry historis. Perbandingan ini membuktikan apakah proses pipeline "
            "eksperimen berhasil meningkatkan akurasi prediksi secara signifikan."
        )

        has_exp_id = "experiment_id" in final_df.columns

        # Load registry early for enriching Terbaik Historis table
        registry_path = results_dir / "best_results_registry.json"
        registry_for_final = {}
        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                registry_for_final = json.load(f)

        configs = final_df["configuration"].unique()
        for config_name in configs:
            cdf = final_df[final_df["configuration"] == config_name].sort_values("RMSE")
            pdf.sub_title(f"Konfigurasi: {config_name}")

            # Use enriched table for Terbaik Historis if registry is available
            if "Historis" in config_name and registry_for_final:
                headers = ["Model", "ID", "Imputasi", "Fitur",
                           "Scaler", "FE", "RMSE", "MAPE", "R2"]
                col_w = [16, 12, 24, 22, 22, 22, 18, 18, 18]
                rows = []
                for _, r in cdf.iterrows():
                    model = r["model"]
                    reg_info = registry_for_final.get(model, {})
                    combo = reg_info.get("combination", {})
                    rows.append([
                        model,
                        str(r.get("experiment_id", "")),
                        _label(combo.get("imputation", "?"))[:12],
                        _label(combo.get("features", "?"))[:12],
                        _label(combo.get("scaler", "?"))[:12],
                        _label(combo.get("feature_engineering", "?"))[:12],
                        f"{r['RMSE']:.4f}",
                        f"{r['MAPE']:.2f}" if pd.notna(r.get("MAPE")) else "-",
                        f"{r['R2']:.4f}" if pd.notna(r.get("R2")) else "-",
                    ])
                pdf.add_table(headers, rows, col_widths=col_w)

                # Add per-model combination detail below the table
                pdf.sub_title("Detail Kombinasi Preprocessing per Model")
                for _, r in cdf.iterrows():
                    model = r["model"]
                    reg_info = registry_for_final.get(model, {})
                    combo = reg_info.get("combination", {})
                    source_phase = reg_info.get("source_phase", "?")
                    source_label = {
                        "focused": "Evaluasi Mendalam (Fase 3)",
                        "tuning": "Optimasi Hyperparameter (Fase 4)",
                        "final": "Perbandingan Final (Fase 5)",
                    }.get(source_phase, source_phase)

                    pdf.set_font("Helvetica", "B", 9)
                    pdf.set_text_color(41, 128, 185)
                    pdf.cell(0, 6, sanitize_text(
                        f"{model}  (ID: {r.get('experiment_id', '?')}  |  "
                        f"RMSE: {r['RMSE']:.4f})"),
                        new_x="LMARGIN", new_y="NEXT")
                    pdf.combo_detail_block(combo)
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.set_text_color(100, 100, 100)
                    pdf.cell(0, 5, sanitize_text(f"Sumber: {source_label}"),
                             new_x="LMARGIN", new_y="NEXT")
                    hp = reg_info.get("hyperparams")
                    if hp:
                        pdf.set_font("Helvetica", "", 8)
                        pdf.set_text_color(80, 80, 80)
                        pdf.cell(0, 5, sanitize_text(f"Hyperparameter: {hp}"),
                                 new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(3)
            else:
                if has_exp_id:
                    headers = ["Model", "ID", "RMSE", "MAPE (%)", "R2"]
                    col_w = [30, 22, 42, 42, 42]
                else:
                    headers = ["Model", "RMSE", "MAPE (%)", "R2"]
                    col_w = [50, 46, 46, 46]
                rows = []
                for _, r in cdf.iterrows():
                    row_data = [r["model"]]
                    if has_exp_id:
                        row_data.append(str(r.get("experiment_id", "")))
                    row_data.extend([
                        f"{r['RMSE']:.4f}",
                        f"{r['MAPE']:.2f}" if pd.notna(r.get("MAPE")) else "-",
                        f"{r['R2']:.4f}" if pd.notna(r.get("R2")) else "-",
                    ])
                    rows.append(row_data)
                pdf.add_table(headers, rows, col_widths=col_w)

        # Show best overall across all configurations
        best_overall = final_df.loc[final_df["RMSE"].idxmin()]
        exp_id_note = ""
        if has_exp_id and pd.notna(best_overall.get("experiment_id")):
            exp_id_note = f" (kombinasi {best_overall['experiment_id']})"
        pdf.body_text(
            f"Secara keseluruhan, model {best_overall['model']} pada konfigurasi "
            f"'{best_overall['configuration']}'{exp_id_note} mencapai performa terbaik "
            f"dengan RMSE {best_overall['RMSE']:.4f} dan R2 {best_overall['R2']:.4f}."
        )

        # Per-model comparison: Baseline vs Current Run vs Registry
        baseline_df = final_df[final_df["configuration"] == "Baseline"]
        current_df = final_df[final_df["configuration"].str.contains("Run Saat Ini", na=False)]
        registry_df = final_df[final_df["configuration"].str.contains("Historis", na=False)]

        if not baseline_df.empty and (not current_df.empty or not registry_df.empty):
            pdf.sub_title("Perbandingan Peningkatan per Model")

            header_parts = ["Model", "Baseline"]
            cw_parts = [30, 28]
            if not current_df.empty:
                header_parts.append("Run Ini")
                cw_parts.append(28)
            if not registry_df.empty:
                header_parts.append("Historis")
                cw_parts.append(28)
            header_parts.extend(["Terbaik", "vs Baseline"])
            cw_parts.extend([28, 28])

            # Pad to fill width
            remaining = 190 - sum(cw_parts)
            if remaining > 0:
                cw_parts[-1] += remaining

            rows = []
            for model in baseline_df["model"].unique():
                bl = baseline_df[baseline_df["model"] == model]
                if bl.empty:
                    continue
                bl_rmse = bl["RMSE"].values[0]
                if pd.isna(bl_rmse):
                    continue

                row_data = [model, f"{bl_rmse:.4f}"]

                # Current run RMSE
                cur_rmse = float("inf")
                if not current_df.empty:
                    cur = current_df[current_df["model"] == model]
                    cur_rmse = cur["RMSE"].values[0] if not cur.empty else float("inf")
                    row_data.append(f"{cur_rmse:.4f}" if cur_rmse != float("inf") else "-")

                # Registry RMSE
                reg_rmse = float("inf")
                if not registry_df.empty:
                    reg = registry_df[registry_df["model"] == model]
                    reg_rmse = reg["RMSE"].values[0] if not reg.empty else float("inf")
                    row_data.append(f"{reg_rmse:.4f}" if reg_rmse != float("inf") else "-")

                # Best among current and registry
                best_opt = min(cur_rmse, reg_rmse)
                if best_opt == float("inf"):
                    row_data.extend(["-", "-"])
                else:
                    diff = bl_rmse - best_opt
                    pct = (diff / bl_rmse * 100) if bl_rmse > 0 else 0
                    row_data.append(f"{best_opt:.4f}")
                    row_data.append(f"{pct:+.1f}%")

                rows.append(row_data)

            if rows:
                pdf.add_table(header_parts, rows, col_widths=cw_parts)

        # Registry vs Current Run detail (if both exist)
        if not current_df.empty and not registry_df.empty:
            pdf.sub_title("Run Saat Ini vs Terbaik Historis")
            pdf.body_text(
                "Tabel berikut menunjukkan apakah eksperimen yang dijalankan "
                "saat ini menghasilkan performa yang lebih baik dibandingkan "
                "hasil terbaik yang pernah tercatat sepanjang seluruh eksperimen."
            )
            headers = ["Model", "Run Ini", "Historis", "Selisih", "Status"]
            rows = []
            for model in current_df["model"].unique():
                cur = current_df[current_df["model"] == model]
                reg = registry_df[registry_df["model"] == model]
                if cur.empty:
                    continue
                cur_rmse = cur["RMSE"].values[0]
                reg_rmse = reg["RMSE"].values[0] if not reg.empty else float("inf")

                if reg_rmse == float("inf"):
                    status = "Baru"
                    diff_str = "-"
                elif cur_rmse < reg_rmse:
                    status = "Lebih Baik"
                    diff_str = f"{cur_rmse - reg_rmse:+.4f}"
                elif cur_rmse == reg_rmse:
                    status = "Sama"
                    diff_str = "0.0000"
                else:
                    status = "Historis Unggul"
                    diff_str = f"{cur_rmse - reg_rmse:+.4f}"

                rows.append([
                    model,
                    f"{cur_rmse:.4f}",
                    f"{reg_rmse:.4f}" if reg_rmse != float("inf") else "-",
                    diff_str,
                    status,
                ])
            if rows:
                pdf.add_table(headers, rows,
                              col_widths=[34, 34, 34, 34, 40])

    # Kesimpulan
    pdf.add_page()
    pdf.section_title("6. Kesimpulan")

    # Load registry for conclusion if available
    registry_path = results_dir / "best_results_registry.json"
    registry_data = {}
    if registry_path.exists():
        import json as _json
        with open(registry_path, encoding="utf-8") as _f:
            registry_data = _json.load(_f)

    if registry_data:
        # Use registry for the definitive "best model" conclusion
        best_model = min(registry_data,
                         key=lambda m: registry_data[m].get("RMSE", float("inf")))
        best_info = registry_data[best_model]
        best_rmse = best_info.get("RMSE", 0)
        best_exp_id = best_info.get("experiment_id", "?")
        best_source = best_info.get("source_phase", "?")
        best_label = best_info.get("label", "")

        source_desc = {
            "focused": "evaluasi mendalam (Fase 3)",
            "tuning": "optimasi hyperparameter (Fase 4)",
            "final": "perbandingan final (Fase 5)",
        }.get(best_source, best_source)

        pdf.body_text(
            f"Berdasarkan seluruh rangkaian eksperimen yang telah dilakukan, "
            f"model {best_model} dengan konfigurasi {best_exp_id} "
            f"({best_label}) merupakan arsitektur prediksi BI-Rate "
            f"yang paling akurat sepanjang keseluruhan eksperimen. "
            f"Hasil terbaik ini diperoleh pada tahap {source_desc}, "
            f"dengan RMSE sebesar {best_rmse:.4f} yang menunjukkan simpangan "
            f"prediksi rata-rata sekitar {best_rmse:.2f} persen poin dari "
            f"nilai aktual BI-Rate."
        )

        # Show registry summary table
        pdf.sub_title("Rekapitulasi Terbaik per Model (Seluruh Eksperimen)")
        headers = ["Model", "ID", "Imputasi", "Fitur", "Scaler", "FE",
                   "RMSE", "R2", "Sumber"]
        rows = []
        for m in sorted(registry_data,
                        key=lambda x: registry_data[x].get("RMSE", float("inf"))):
            info = registry_data[m]
            combo = info.get("combination", {})
            source_label = {
                "focused": "Fase 3",
                "tuning": "Fase 4 (Tuned)",
                "final": "Fase 5",
            }.get(info.get("source_phase", ""), info.get("source_phase", "?"))
            rows.append([
                m,
                info.get("experiment_id", "?"),
                _label(combo.get("imputation", "?"))[:12],
                _label(combo.get("features", "?"))[:12],
                _label(combo.get("scaler", "?"))[:12],
                _label(combo.get("feature_engineering", "?"))[:12],
                f"{info.get('RMSE', 0):.4f}",
                f"{info.get('R2', 0):.4f}" if info.get("R2") is not None else "-",
                source_label,
            ])
        pdf.add_table(headers, rows,
                      col_widths=[18, 14, 22, 22, 22, 22, 20, 20, 28])

        # Detail konfigurasi per model
        pdf.sub_title("Detail Konfigurasi Terbaik per Model")
        for m in sorted(registry_data,
                        key=lambda x: registry_data[x].get("RMSE", float("inf"))):
            info = registry_data[m]
            combo = info.get("combination", {})
            hp = info.get("hyperparams")
            source_label = {
                "focused": "Evaluasi Mendalam (Fase 3)",
                "tuning": "Optimasi Hyperparameter (Fase 4)",
                "final": "Perbandingan Final (Fase 5)",
            }.get(info.get("source_phase", ""), info.get("source_phase", "?"))

            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(41, 128, 185)
            pdf.cell(0, 6, sanitize_text(
                f"{m}  (ID: {info.get('experiment_id', '?')}  |  "
                f"RMSE: {info.get('RMSE', 0):.4f}  |  "
                f"MAPE: {info.get('MAPE', 0):.2f}%  |  "
                f"R2: {info.get('R2', 0):.4f})"),
                new_x="LMARGIN", new_y="NEXT")
            pdf.combo_detail_block(combo)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 5, sanitize_text(f"Sumber: {source_label}"),
                     new_x="LMARGIN", new_y="NEXT")
            if hp:
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(0, 5, sanitize_text(f"Hyperparameter: {hp}"),
                         new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

    elif not final_df.empty:
        # Fallback: no registry, use final_df as before
        best = final_df.loc[final_df["RMSE"].idxmin()]
        pdf.body_text(
            f"Berdasarkan seluruh rangkaian eksperimen yang telah dilakukan, "
            f"dapat disimpulkan bahwa model {best['model']} dengan konfigurasi "
            f"'{best['configuration']}' merupakan arsitektur prediksi BI-Rate "
            f"yang paling akurat. Model ini mencapai tingkat error (RMSE) "
            f"sebesar {best['RMSE']:.4f}, yang menunjukkan simpangan prediksi "
            f"rata-rata sekitar {best['RMSE']:.2f} persen poin dari nilai aktual BI-Rate."
        )

    if factor_analysis:
        importance = factor_analysis.get("importance", {})
        if importance:
            imp_sorted = sorted(importance.items(),
                                key=lambda x: x[1]["range"], reverse=True)
            top_factor = imp_sorted[0]
            pdf.body_text(
                f"Dari analisis faktor, komponen '{top_factor[0].replace('_',' ')}' "
                f"terbukti memiliki pengaruh terbesar terhadap kualitas prediksi. "
                f"Level terbaik untuk faktor ini adalah "
                f"'{_label(top_factor[1]['best_level'])}', yang secara konsisten "
                f"memberikan RMSE lebih rendah dibandingkan alternatif lainnya."
            )
    # Catatan Metodologi
    pdf.add_page()
    pdf.section_title("Catatan Metodologi")
    pdf.body_text(
        "Seluruh eksperimen menggunakan chronological time-series split "
        "(train/validation/test) tanpa shuffling untuk mencegah data leakage. "
        "Scaler hanya di-fit pada training set kemudian diterapkan pada "
        "validation dan test set."
    )
    pdf.body_text(
        "Hyperparameter tuning menggunakan Bayesian Optimization (Optuna) "
        "dengan expanding-window cross-validation yang menjamin evaluasi "
        "selalu dilakukan pada data masa depan relatif terhadap data training."
    )
    pdf.body_text(
        "Metrik evaluasi yang digunakan:\n"
        "- RMSE (Root Mean Squared Error): mengukur rata-rata simpangan "
        "prediksi, sensitif terhadap outlier\n"
        "- MAPE (Mean Absolute Percentage Error): persentase error "
        "rata-rata, mudah diinterpretasi\n"
        "- R2 (R-squared): proporsi variansi target yang berhasil "
        "dijelaskan oleh model"
    )

    pdf.output(str(output_path))
    print(f"Laporan PDF eksperimen berhasil disimpan ke: {output_path}")
    return str(output_path)
