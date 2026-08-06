"""Runner eksperimen faktorial prediksi BI-Rate (lima fase)."""

import json
import time
import warnings
import gc
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from src.modeling.config import DATASET_PATH, TARGET_COLUMN, SEQUENCE_LENGTH
from src.modeling.data_prep import (
    load_dataset, handle_missing_values, time_series_split,
    scale_features, create_sequences,
)
from src.modeling.metrics import evaluate_model, evaluate_quiet
from src.modeling.traditional import ARIMAModel, VARModel
from src.modeling.machine_learning import RandomForestModel, XGBoostModel

from src.experiments.feature_engineering import (
    apply_feature_engineering, select_features_mutual_info,
    add_target_lag_features,
)
from src.experiments.hyperparameter_tuner import tune_model
from src.experiments.scenario_config import (
    generate_experiment_grid, build_experiment_config,
    SCREENING_MODELS, ALL_MODELS, DEFAULT_HYPERPARAMS,
    TUNING_CONFIGS, ALL_FEATURES,
)
from src.experiments.best_registry import (
    load_registry, save_registry, update_if_better, prune_model_files,
    REGISTRY_FILENAME,
)

# seed ensembel NN; prediksi dirata-ratakan agar stabil antar-run
DL_ENSEMBLE_SEEDS = [42, 43, 44]


class FactorialExperimentRunner:
    """Orkestrasi lima fase eksperimen faktorial (bisa full pipeline
    via run_pipeline() atau per-fase: screening/analyze/focused/tuning/final)."""

    def __init__(self, output_dir=None, dataset_path=None):
        self.output_dir = Path(output_dir or "results/experiments")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = dataset_path or str(DATASET_PATH)
        self._raw_df = None

        # Results storage
        self.screening_results = pd.DataFrame()   # Phase 1
        self.factor_analysis = {}                  # Phase 2
        self.focused_results = pd.DataFrame()      # Phase 3
        self.tuning_results = {}                   # Phase 4
        self.final_results = pd.DataFrame()        # Phase 5

        # Persistent best results registry
        self.registry_path = self.output_dir / REGISTRY_FILENAME
        self.best_registry = load_registry(self.registry_path)

    def _load_raw_data(self):
        if self._raw_df is None:
            self._raw_df = load_dataset(self.dataset_path)
        return self._raw_df.copy()

    # Full Pipeline
    def run_pipeline(self, top_focused=10, top_tuning=3, skip_dl=False):
        """Jalankan kelima fase berurutan."""
        print("Memulai Pipeline Eksperimen BI-Rate")

        self.run_screening()
        self.analyze_factors()
        self.run_focused(top_n=top_focused, skip_dl=skip_dl)
        self.run_tuning(top_n=top_tuning)
        self.run_final()

        # Sisakan hanya model terbaik per algoritma di folder models/
        removed = prune_model_files(self.best_registry)
        if removed:
            print(f"  Pembersihan models/: {removed} berkas non-terbaik dihapus.")

        print(f"Pipeline selesai. Hasil disimpan di: {self.output_dir}")

    # Phase 1: Screening
    def run_screening(self, models=None):
        """Phase 1: screening 384 kombinasi dengan RF + XGBoost (768 evaluasi)."""
        models = models or SCREENING_MODELS
        grid = generate_experiment_grid()

        phase_dir = self.output_dir / "screening"
        phase_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Fase 1: Screening Model ---")
        print(f"Evaluasi {len(grid)} kombinasi pada {len(models)} model "
              f"(Total: {len(grid) * len(models)} iterasi)")

        all_rows = []
        start = time.time()

        for i, combo in enumerate(grid, 1):
            if i % 20 == 0 or i == 1:
                try:
                    import psutil
                    mem = psutil.Process().memory_info().rss / 1024 / 1024
                    print(f"    [DIAG] Iteration {i}: RAM RSS = {mem:.0f}MB")
                except ImportError:
                    pass
                print(f"\n  [{i}/{len(grid)}] {combo['id']}: {combo['label'][:60]}...")

            config = build_experiment_config(combo)
            config.setdefault("_factors", {})["_exp_id"] = combo["id"]
            results = self._run_experiment(config, models, persist=False)

            for model_name, metrics in results.items():
                if isinstance(metrics, dict) and "RMSE" in metrics:
                    row = {
                        "experiment_id": combo["id"],
                        "imputation": combo["imputation"],
                        "features": combo["features"],
                        "scaler": combo["scaler"],
                        "feature_engineering": combo["feature_engineering"],
                        "model": model_name,
                        "RMSE": metrics["RMSE"],
                        "MAPE": metrics.get("MAPE"),
                        "R2": metrics.get("R2"),
                        "RMSE_val": metrics.get("RMSE_val"),
                        "MAPE_val": metrics.get("MAPE_val"),
                        "R2_val": metrics.get("R2_val"),
                        "model_path": metrics.get("model_path"),
                        "label": combo["label"],
                    }
                    all_rows.append(row)

        elapsed = time.time() - start
        self.screening_results = pd.DataFrame(all_rows)

        if self.screening_results.empty:
            print("  No valid results!")
            return

        # Save
        self.screening_results.to_csv(phase_dir / "screening_results.csv", index=False)

        # Summary
        avg_per_exp = (self.screening_results.groupby("experiment_id")["RMSE"]
                       .mean().reset_index().rename(columns={"RMSE": "avg_RMSE"}))
        avg_per_exp = avg_per_exp.sort_values("avg_RMSE")
        avg_per_exp.to_csv(phase_dir / "screening_ranking.csv", index=False)

        print(f"\n  Screening selesai dalam waktu {elapsed:.1f} detik.")
        print(f"\n  10 Kombinasi Terbaik (berdasarkan rata-rata RMSE):")
        print(f"  {'Rank':<6} {'ID':<8} {'Rata-rata RMSE':<15} {'Keterangan'}")
        print(f"  {'-'*75}")

        for rank, (_, row) in enumerate(avg_per_exp.head(10).iterrows(), 1):
            combo = next(c for c in grid if c["id"] == row["experiment_id"])
            print(f"  {rank:<6} {row['experiment_id']:<8} "
                  f"{row['avg_RMSE']:<12.4f} {combo['label'][:50]}")

        # Save grid metadata
        with open(phase_dir / "experiment_grid.json", "w") as f:
            json.dump(grid, f, indent=2)

        self._generate_screening_report(phase_dir, grid, avg_per_exp, elapsed)

    # Phase 2: Factor Analysis
    def analyze_factors(self):
        """Phase 2: pengaruh marginal tiap faktor + interaksi dua-faktor."""
        if self.screening_results.empty:
            self._load_screening_results()
        if self.screening_results.empty:
            print("  No screening results to analyze!")
            return

        phase_dir = self.output_dir / "analysis"
        phase_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Fase 2: Analisis Faktor ---")

        df = self.screening_results.copy()
        factors = ["imputation", "features", "scaler", "feature_engineering"]

        print(f"\n  Dampak Marginal (Rata-rata performa berdasarkan fitur individu):")
        marginal_tables = {}

        for factor in factors:
            marginal = (df.groupby(factor)["RMSE"]
                        .agg(["mean", "std", "min", "max", "count"])
                        .sort_values("mean"))
            marginal_tables[factor] = marginal

            print(f"\n  {factor.upper()}:")
            print(f"  {'Level':<20} {'Mean RMSE':<12} {'Std':<10} {'Min':<10} "
                  f"{'Max':<10} {'N'}")
            print(f"  {'-'*72}")
            for level, row in marginal.iterrows():
                print(f"  {str(level):<20} {row['mean']:<12.4f} "
                      f"{row['std']:<10.4f} {row['min']:<10.4f} "
                      f"{row['max']:<10.4f} {int(row['count'])}")

            marginal.to_csv(phase_dir / f"marginal_{factor}.csv")

        # 2b. Factor Importance
        print(f"\n  Tingkat Signifikansi Faktor (Berdasarkan simpangan rata-rata):")
        importance = {}
        for factor in factors:
            means = marginal_tables[factor]["mean"]
            importance[factor] = {
                "range": means.max() - means.min(),
                "best_level": means.idxmin(),
                "worst_level": means.idxmax(),
                "best_mean": means.min(),
                "worst_mean": means.max(),
            }

        imp_sorted = sorted(importance.items(), key=lambda x: x[1]["range"],
                            reverse=True)
        print(f"\n  {'Rank':<6} {'Factor':<22} {'Range':<10} "
              f"{'Best Level':<18} {'Worst Level'}")
        print(f"  {'-'*75}")
        for rank, (factor, info) in enumerate(imp_sorted, 1):
            print(f"  {rank:<6} {factor:<22} {info['range']:<10.4f} "
                  f"{info['best_level']:<18} {info['worst_level']}")

        # 2c. Interaction Effects (pairwise)
        print(f"\n  Efek Interaksi (Analisis rata-rata RMSE antar pasangan faktor):")
        interaction_tables = {}

        for i, f1 in enumerate(factors):
            for f2 in factors[i+1:]:
                pivot = df.pivot_table(values="RMSE", index=f1, columns=f2,
                                       aggfunc="mean")
                interaction_tables[f"{f1}_x_{f2}"] = pivot

                # Measure interaction strength: std of cell means
                cell_means = pivot.values.flatten()
                cell_means = cell_means[~np.isnan(cell_means)]
                interaction_strength = np.std(cell_means)

                print(f"\n  {f1} × {f2} (interaction strength = {interaction_strength:.4f}):")
                print(f"  {pivot.to_string(float_format=lambda x: f'{x:.4f}')}")

                pivot.to_csv(phase_dir / f"interaction_{f1}_x_{f2}.csv")

        # 2d. Per-model analysis
        print(f"\n  Analisis Variansi Per-Model:")
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            print(f"\n  Model: {model}")
            for factor in factors:
                best = model_df.groupby(factor)["RMSE"].mean().idxmin()
                best_val = model_df.groupby(factor)["RMSE"].mean().min()
                print(f"    Best {factor:<22}: {best} (RMSE = {best_val:.4f})")

        # Save analysis
        self.factor_analysis = {
            "marginal_tables": {k: v.to_dict() for k, v in marginal_tables.items()},
            "importance": importance,
        }

        with open(phase_dir / "factor_analysis.json", "w") as f:
            json.dump(self.factor_analysis, f, indent=2, default=str)

        self._generate_analysis_report(phase_dir, marginal_tables,
                                        importance, interaction_tables)

    # Phase 3: Focused Evaluation
    def run_focused(self, top_n=10, skip_dl=False):
        """Phase 3: evaluasi top-N kombinasi screening dengan semua model."""
        if self.screening_results.empty:
            self._load_screening_results()

        phase_dir = self.output_dir / "focused"
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Get top-N combinations
        avg_per_exp = (self.screening_results.groupby("experiment_id")["RMSE"]
                       .mean().sort_values().head(top_n))
        top_ids = avg_per_exp.index.tolist()

        # Load grid metadata
        grid_path = self.output_dir / "screening" / "experiment_grid.json"
        with open(grid_path) as f:
            grid = json.load(f)
        grid_dict = {c["id"]: c for c in grid}

        models = [m for m in ALL_MODELS if not (skip_dl and m in ("LSTM", "BiLSTM"))]

        print(f"\n--- Fase 3: Evaluasi Mendalam (Focused Evaluation) ---")
        print(f"Menguji Top-{top_n} kombinasi menggunakan seluruh algoritma ({len(models)} model)")

        all_rows = []
        start = time.time()

        for i, exp_id in enumerate(top_ids, 1):
            combo = grid_dict[exp_id]
            print(f"\n  [{i}/{len(top_ids)}] {exp_id}: {combo['label'][:60]}...")

            config = build_experiment_config(combo)
            config.setdefault("_factors", {})["_exp_id"] = exp_id
            results = self._run_experiment(config, models)

            for model_name, metrics in results.items():
                if isinstance(metrics, dict) and "RMSE" in metrics:
                    row = {
                        "experiment_id": exp_id,
                        "imputation": combo["imputation"],
                        "features": combo["features"],
                        "scaler": combo["scaler"],
                        "feature_engineering": combo["feature_engineering"],
                        "model": model_name,
                        "RMSE": metrics["RMSE"],
                        "MAPE": metrics.get("MAPE"),
                        "R2": metrics.get("R2"),
                        "RMSE_val": metrics.get("RMSE_val"),
                        "MAPE_val": metrics.get("MAPE_val"),
                        "R2_val": metrics.get("R2_val"),
                        "model_path": metrics.get("model_path"),
                        "label": combo["label"],
                    }
                    all_rows.append(row)

        elapsed = time.time() - start
        self.focused_results = pd.DataFrame(all_rows)
        self.focused_results.to_csv(phase_dir / "focused_results.csv", index=False)

        # Best overall
        if not self.focused_results.empty:
            best_row = self.focused_results.loc[self.focused_results["RMSE"].idxmin()]

            print(f"\n  Fase evaluasi mendalam tuntas dalam {elapsed:.1f} detik.")
            print(f"\n  * Konfigurasi Paling Optimal Keseluruhan:")
            print(f"    Combination: {best_row['experiment_id']} — {best_row['label']}")
            print(f"    Model:       {best_row['model']}")
            print(f"    RMSE:        {best_row['RMSE']:.4f}" if pd.notna(best_row['RMSE']) else "    RMSE:        N/A")
            print(f"    MAPE:        {best_row['MAPE']:.2f}%" if pd.notna(best_row['MAPE']) else "    MAPE:        N/A")
            print(f"    R²:          {best_row['R2']:.4f}" if pd.notna(best_row['R2']) else "    R²:          N/A")

            # Ranking table
            print(f"\n  TOP RESULTS:")
            top_results = self.focused_results.nsmallest(15, "RMSE")
            print(f"  {'Rank':<6} {'ID':<8} {'Model':<12} {'RMSE':<10} "
                  f"{'MAPE':<10} {'R²':<10}")
            print(f"  {'-'*60}")
            for rank, (_, r) in enumerate(top_results.iterrows(), 1):
                rmse_s = f"{r['RMSE']:<10.4f}" if pd.notna(r['RMSE']) else f"{'N/A':<10}"
                mape_s = f"{r['MAPE']:<10.2f}" if pd.notna(r['MAPE']) else f"{'N/A':<10}"
                r2_s = f"{r['R2']:<10.4f}" if pd.notna(r['R2']) else f"{'N/A':<10}"
                print(f"  {rank:<6} {r['experiment_id']:<8} {r['model']:<12} "
                      f"{rmse_s} {mape_s} {r2_s}")

            # Update best registry per model
            self._update_registry_from_focused()

        self._generate_focused_report(phase_dir, elapsed)

    # Phase 4: Hyperparameter Tuning
    def run_tuning(self, top_n=3):
        """
        Phase 4: Optimasi hyperparameter pada top-N kombinasi terbaik.
        """
        if self.focused_results.empty:
            self._load_focused_results()

        phase_dir = self.output_dir / "tuning"
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Get top-N unique combinations
        avg = (self.focused_results.groupby("experiment_id")["RMSE"]
               .mean().sort_values().head(top_n))
        top_ids = avg.index.tolist()

        grid_path = self.output_dir / "screening" / "experiment_grid.json"
        with open(grid_path) as f:
            grid_dict = {c["id"]: c for c in json.load(f)}

        print(f"\n--- Fase 4: Optimasi Hyperparameter (Tuning) ---")
        print(f"Fokus pada varian terbaik dari {top_n} kombinasi teratas")

        all_rows = []
        start = time.time()

        for exp_id in top_ids:
            combo = grid_dict[exp_id]
            print(f"\n  {exp_id}: {combo['label'][:60]}...")

            config = build_experiment_config(combo)
            config.setdefault("_factors", {})["_exp_id"] = exp_id

            # Get best models for this combination from focused results
            combo_results = self.focused_results[
                self.focused_results["experiment_id"] == exp_id
            ]
            best_models = combo_results.nsmallest(3, "RMSE")["model"].tolist()

            for model_name in best_models:
                if model_name not in TUNING_CONFIGS:
                    continue

                print(f"    Tuning {model_name}...")
                result = self._run_tuned_experiment(
                    config, model_name, TUNING_CONFIGS[model_name]
                )

                if result:
                    row = {
                        "experiment_id": exp_id,
                        "model": model_name,
                        "RMSE_default": combo_results[
                            combo_results["model"] == model_name
                        ]["RMSE"].values[0] if len(combo_results[
                            combo_results["model"] == model_name
                        ]) > 0 else None,
                        "RMSE_tuned": result.get("RMSE"),
                        "MAPE_tuned": result.get("MAPE"),
                        "R2_tuned": result.get("R2"),
                        "model_path": result.get("model_path"),
                        "best_params": str(result.get("best_params", {})),
                        "label": combo["label"],
                    }
                    all_rows.append(row)

        elapsed = time.time() - start
        tuning_df = pd.DataFrame(all_rows)

        if not tuning_df.empty:
            tuning_df.to_csv(phase_dir / "tuning_results.csv", index=False)

            print(f"\n  Optimasi hyperparameter selesai dikerjakan dalam {elapsed:.1f} detik.")
            print(f"\n  RINGKASAN HASIL:")
            for _, r in tuning_df.iterrows():
                default = r.get("RMSE_default")
                tuned = r.get("RMSE_tuned")
                if default and tuned and default != 0:
                    change = ((tuned - default) / default) * 100
                    status = "membaik" if tuned < default else "TIDAK MEMBAIK"
                    print(f"    {r['experiment_id']} / {r['model']}: "
                          f"{default:.4f} -> {tuned:.4f} ({change:+.1f}%) [{status}]")

            # Update registry with tuning results (guardrail: only if improved)
            self._update_registry_from_tuning(tuning_df)

        self.tuning_results = tuning_df

    # Phase 5: Final Comparison
    def run_final(self):
        """
        Phase 5: Perbandingan akhir — baseline vs optimized vs historical best.

        Menjalankan model pada baseline dan konfigurasi terbaik dari focused,
        kemudian membandingkan hasilnya dengan registry historis terbaik.
        """
        phase_dir = self.output_dir / "final"
        phase_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Fase 5: Evaluasi Final ---")

        if self.focused_results.empty:
            self._load_focused_results()

        # Reload registry to get latest state
        self.best_registry = load_registry(self.registry_path)

        print(f"\n  Mengevaluasi konfigurasi baseline (interpolate + all + standard + none)...")
        baseline_combo = {
            "id": "BASELINE",
            "imputation": "interpolate",
            "features": "all",
            "scaler": "standard",
            "feature_engineering": "none",
        }
        baseline_config = build_experiment_config(baseline_combo)
        baseline_config.setdefault("_factors", {})["_exp_id"] = "BASELINE"
        baseline_results = self._run_experiment(baseline_config, ALL_MODELS)

        rows = []

        for model, metrics in baseline_results.items():
            if isinstance(metrics, dict) and "RMSE" in metrics:
                rows.append({
                    "configuration": "Baseline",
                    "experiment_id": "BASELINE",
                    "source": "current_run",
                    "model": model,
                    "RMSE": metrics["RMSE"],
                    "MAPE": metrics.get("MAPE"),
                    "R2": metrics.get("R2"),
                })

        if not self.focused_results.empty:
            best_focused = self.focused_results.loc[
                self.focused_results["RMSE"].idxmin()
            ]
            grid_path = self.output_dir / "screening" / "experiment_grid.json"
            if grid_path.exists():
                with open(grid_path) as f:
                    grid_dict = {c["id"]: c for c in json.load(f)}
                best_id = best_focused["experiment_id"]
                if best_id in grid_dict:
                    best_combo = grid_dict[best_id]
                    print(f"\n  Mengevaluasi konfigurasi terbaik: {best_id} ({best_combo['label'][:60]}...)")
                    best_config = build_experiment_config(best_combo)
                    best_config.setdefault("_factors", {})["_exp_id"] = best_id
                    best_results = self._run_experiment(best_config, ALL_MODELS)
                    for model, metrics in best_results.items():
                        if isinstance(metrics, dict) and "RMSE" in metrics:
                            rows.append({
                                "configuration": f"Run Saat Ini ({best_id})",
                                "experiment_id": best_id,
                                "source": "current_run",
                                "model": model,
                                "RMSE": metrics["RMSE"],
                                "MAPE": metrics.get("MAPE"),
                                "R2": metrics.get("R2"),
                            })

                            # Update registry if this run is better
                            update_if_better(self.best_registry, model, {
                                "experiment_id": best_id,
                                "RMSE": metrics["RMSE"],
                                "MAPE": metrics.get("MAPE"),
                                "R2": metrics.get("R2"),
                                "source_phase": "final",
                                "combination": {
                                    "imputation": best_combo.get("imputation"),
                                    "features": best_combo.get("features"),
                                    "scaler": best_combo.get("scaler"),
                                    "feature_engineering": best_combo.get("feature_engineering"),
                                },
                                "label": best_combo.get("label", ""),
                                "model_path": metrics.get("model_path"),
                            })

        if self.best_registry:
            print(f"\n  Memuat hasil terbaik historis dari registry...")
            for model, info in self.best_registry.items():
                rmse = info.get("RMSE")
                if rmse is not None and rmse != float("inf"):
                    rows.append({
                        "configuration": "Terbaik Historis (Registry)",
                        "experiment_id": info.get("experiment_id", "?"),
                        "source": f"{info.get('source_phase', '?')}",
                        "model": model,
                        "RMSE": rmse,
                        "MAPE": info.get("MAPE"),
                        "R2": info.get("R2"),
                    })

        # Save registry updates
        save_registry(self.best_registry, self.registry_path)

        self.final_results = pd.DataFrame(rows)
        self.final_results.to_csv(phase_dir / "final_comparison.csv", index=False)

        if not self.final_results.empty:
            print(f"\n  HASIL PERBANDINGAN FINAL:")
            pivot = self.final_results.pivot_table(
                values="RMSE", index="model", columns="configuration",
                aggfunc="first"
            )
            print(f"\n{pivot.to_string(float_format=lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')}")

            valid_rmse = self.final_results[self.final_results["RMSE"].notna()]
            if not valid_rmse.empty:
                overall_best = valid_rmse.loc[valid_rmse["RMSE"].idxmin()]
                rmse_str = f"{overall_best['RMSE']:.4f}" if pd.notna(overall_best['RMSE']) else "N/A"
                print(f"\n  * PALING OPTIMAL: {overall_best['configuration']} / "
                      f"{overall_best['model']} (RMSE = {rmse_str})")

            # Show registry vs current run comparison
            self._print_registry_comparison()

        self._generate_final_report(phase_dir)

    # Core Experiment Execution
    def _run_experiment(self, config, models, persist=True):
        """Jalankan satu kombinasi; persist=False (screening) tanpa simpan model."""
        try:
            df = self._load_raw_data()

            # Imputation
            extra_kw = {}
            if config["missing_method"] == "knn":
                extra_kw["knn_neighbors"] = config.get("knn_neighbors", 5)
            df = handle_missing_values(df, method=config["missing_method"], **extra_kw)

            # Feature engineering
            fe_config = config.get("feature_engineering")
            new_features = []
            if fe_config and isinstance(fe_config, dict):
                fe_type = fe_config.get("type")
                if fe_type == "target_lag":
                    df, new_features = add_target_lag_features(
                        df, TARGET_COLUMN, fe_config["lags"])
                elif fe_type in ("lag", "rolling", "diff", "combined",
                                "interaction", "calendar", "policy_reaction"):
                    df, new_features = apply_feature_engineering(
                        df, fe_config, TARGET_COLUMN)

            # Features
            feature_cols = list(config.get("feature_columns", ALL_FEATURES))

            # Mutual info selection
            if config.get("selection_method") == "mutual_info":
                feature_cols, _ = select_features_mutual_info(
                    df, TARGET_COLUMN, feature_cols,
                    n_features=config.get("n_select", 7))

            var_feature_cols = feature_cols.copy()
            feature_cols = feature_cols + new_features
            feature_cols = [c for c in feature_cols if c in df.columns]
            if not feature_cols:
                return {"error": "No valid features"}

            # Split
            X_train, X_val, X_test, y_train, y_val, y_test = time_series_split(
                df, TARGET_COLUMN, feature_cols)

            # Scale
            X_tr, X_va, X_te, scaler = scale_features(
                X_train, X_val, X_test, scaler_type=config["scaler"])

            # VAR needs original DataFrame (without manual lag features)
            train_end = len(y_train) + len(y_val)
            var_df = df[var_feature_cols + [TARGET_COLUMN]].iloc[:train_end]

            # Train models
            results = {}
            exp_id = config.get("_factors", {}).get("_exp_id", "") if persist else ""
            for model_name in models:
                # Resume: skip if checkpoint AND model file exist
                if exp_id:
                    ckpt = self.output_dir / "checkpoints" / f"{exp_id}_{model_name}.json"
                    ext = ".keras" if model_name in ("LSTM", "BiLSTM") else ".pkl"
                    expected_model_path = Path("models") / f"{exp_id}_{model_name}{ext}"
                    
                    if ckpt.exists():
                        import json as _json
                        try:
                            with open(ckpt) as _f:
                                cached = _json.load(_f)
                        except Exception:
                            cached = {}
                        
                        cached_model_path = cached.get("model_path")
                        if cached_model_path and Path(cached_model_path).exists() and expected_model_path.exists():
                            results[model_name] = cached
                            print(f"    [RESUME] {exp_id}/{model_name} skipped (checkpoint and model exist)")
                            continue
                        else:
                            print(f"    [RETRAIN] {exp_id}/{model_name} (checkpoint found but model file missing/invalid)")

                try:
                    hp = config.get("hyperparams", DEFAULT_HYPERPARAMS)
                    model_hp = hp.get(model_name, {}) if isinstance(hp, dict) else {}
                    res = self._train_and_evaluate(
                        model_name, model_hp,
                        X_tr, X_va, X_te, y_train, y_val, y_test, var_df, exp_id=exp_id)
                    if res:
                        results[model_name] = res
                        # Checkpoint: save immediately
                        if exp_id:
                            ckpt_dir = self.output_dir / "checkpoints"
                            ckpt_dir.mkdir(parents=True, exist_ok=True)
                            ckpt_path = ckpt_dir / f"{exp_id}_{model_name}.json"
                            with open(ckpt_path, "w") as _f:
                                json.dump(res, _f, indent=2, default=str)
                            print(f"    [CHECKPOINT] {exp_id}/{model_name} saved")
                except Exception as e:
                    results[model_name] = {
                        "RMSE": float("inf"), "MAPE": float("inf"),
                        "R2": float("-inf"), "error": str(e),
                    }

            return results

        except Exception as e:
            return {"error": str(e)}

    def _run_tuned_experiment(self, config, model_name, tuning_config):
        """Run experiment with hyperparameter tuning for one model."""
        try:
            df = self._load_raw_data()
            extra_kw = {}
            if config["missing_method"] == "knn":
                extra_kw["knn_neighbors"] = config.get("knn_neighbors", 5)
            df = handle_missing_values(df, method=config["missing_method"], **extra_kw)

            fe_config = config.get("feature_engineering")
            new_features = []
            if fe_config and isinstance(fe_config, dict):
                fe_type = fe_config.get("type")
                if fe_type == "target_lag":
                    df, new_features = add_target_lag_features(
                        df, TARGET_COLUMN, fe_config["lags"])
                elif fe_type in ("lag", "rolling", "diff", "combined",
                                "interaction", "calendar", "policy_reaction"):
                    df, new_features = apply_feature_engineering(
                        df, fe_config, TARGET_COLUMN)

            feature_cols = list(config.get("feature_columns", ALL_FEATURES))
            if config.get("selection_method") == "mutual_info":
                feature_cols, _ = select_features_mutual_info(
                    df, TARGET_COLUMN, feature_cols,
                    n_features=config.get("n_select", 7))

            var_feature_cols = feature_cols.copy()
            feature_cols = feature_cols + new_features
            feature_cols = [c for c in feature_cols if c in df.columns]

            X_train, X_val, X_test, y_train, y_val, y_test = time_series_split(
                df, TARGET_COLUMN, feature_cols)
            X_tr, X_va, X_te, scaler = scale_features(
                X_train, X_val, X_test, scaler_type=config["scaler"])

            train_end = len(y_train) + len(y_val)
            var_df = df[var_feature_cols + [TARGET_COLUMN]].iloc[:train_end]

            # Tune
            X_tune = np.vstack([X_tr, X_va])
            y_tune = np.concatenate([y_train, y_val])

            best_params, _ = tune_model(
                model_name, X_tune, y_tune, tuning_config, train_df=var_df)

            if best_params is None:
                return None

            # Evaluate with best params
            exp_id = config.get("_factors", {}).get("_exp_id", "")
            result = self._train_and_evaluate(
                model_name, best_params,
                X_tr, X_va, X_te, y_train, y_val, y_test, var_df,
                exp_id=f"{exp_id}_tuned" if exp_id else None)

            if result:
                result["best_params"] = best_params
            return result

        except Exception as e:
            print(f"    Tuning error: {e}")
            return None

    # Model Training
    def _save_model_helper(self, model, model_name, exp_id, metrics):
        if not exp_id or metrics is None:
            return metrics
        try:
            ext = ".keras" if model_name in ("LSTM", "BiLSTM") else ".pkl"
            model_dir = Path("models")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path_str = str(model_dir / f"{exp_id}_{model_name}{ext}")
            
            model.save(model_path_str)
            metrics["model_path"] = model_path_str
            print(f"    [SAVED MODEL] {model_name} saved to {model_path_str}")
        except Exception as e:
            print(f"    [SAVE MODEL WARNING] Failed to save {model_name}: {e}")
        return metrics

    @staticmethod
    def _attach_val(metrics, y_val_true, y_val_pred):
        """Sisipkan metrik segmen VALIDASI sebagai kolom terpisah.

        Kolom `RMSE`/`MAPE`/`R2` sengaja tidak disentuh: seluruh registry dan
        angka yang sudah dilaporkan di dokumen mengacu pada segmen uji. Kolom
        `RMSE_val`/`MAPE_val`/`R2_val` ditambahkan agar pemeringkatan kombinasi
        dapat diperiksa ulang tanpa menyentuh himpunan uji (lihat modul
        src/analysis/selection_bias.py).
        """
        if metrics is None:
            return None
        if y_val_true is None or y_val_pred is None or len(y_val_true) == 0:
            return metrics
        v = evaluate_quiet(np.asarray(y_val_true, float),
                           np.asarray(y_val_pred, float))
        metrics["RMSE_val"] = v["RMSE"]
        metrics["MAPE_val"] = v["MAPE"]
        metrics["R2_val"] = v["R2"]
        return metrics

    def _train_and_evaluate(self, model_name, params, X_tr, X_va, X_te,
                            y_train, y_val, y_test, var_df=None, exp_id=None):
        pdict = params if isinstance(params, dict) else {}
        metrics = None

        if model_name == "ARIMA":
            order = pdict.get("order", (1, 1, 1))
            model = ARIMAModel(order=order)
            model.fit(y_train)
            preds = model.predict(len(y_test))
            metrics = evaluate_model(y_test, preds, "ARIMA")
            metrics = self._attach_val(metrics, y_val, model.predict(len(y_val)))
            if metrics:
                metrics = self._save_model_helper(model, model_name, exp_id, metrics)

        elif model_name == "VAR":
            if var_df is None:
                return None
            var = VARModel(maxlags=pdict.get("maxlags", 3),
                           ic=pdict.get("ic", "aic"))
            var.fit(var_df, TARGET_COLUMN)
            lag = var.fitted_model.k_ar
            last = var_df.select_dtypes(include=[np.number]).dropna().values[-lag:]
            preds = var.predict(len(y_test), last)
            metrics = evaluate_model(y_test, preds, "VAR")
            # VAR dipasang pada var_df yang sudah mencakup segmen validasi,
            # jadi metrik validasi akan bersifat dalam-sampel dan tidak
            # dilaporkan.
            if metrics:
                metrics = self._save_model_helper(var, model_name, exp_id, metrics)

        elif model_name == "RF":
            rf_p = {k: v for k, v in pdict.items() if k in
                    ["n_estimators", "max_depth", "min_samples_split",
                     "min_samples_leaf", "max_features", "random_state"]}
            rf_p.setdefault("random_state", 42)
            model = RandomForestModel(**rf_p)
            model.fit(X_tr, y_train)
            metrics = evaluate_model(y_test, model.predict(X_te), "RF")
            metrics = self._attach_val(metrics, y_val, model.predict(X_va))
            if metrics:
                metrics = self._save_model_helper(model, model_name, exp_id, metrics)

        elif model_name == "XGBoost":
            xgb_p = {k: v for k, v in pdict.items() if k in
                     ["n_estimators", "max_depth", "learning_rate", "subsample",
                      "colsample_bytree", "reg_alpha", "reg_lambda",
                      "min_child_weight", "random_state"]}
            xgb_p.setdefault("random_state", 42)
            model = XGBoostModel(**xgb_p)
            model.fit(X_tr, y_train, X_va, y_val)
            metrics = evaluate_model(y_test, model.predict(X_te), "XGBoost")
            # Catatan: segmen validasi juga dipakai sebagai set early stopping,
            # jadi RMSE_val XGBoost sedikit optimistis (tidak sepenuhnya
            # keluar-sampel). RF tidak memakai validasi saat pelatihan.
            metrics = self._attach_val(metrics, y_val, model.predict(X_va))
            if metrics:
                metrics = self._save_model_helper(model, model_name, exp_id, metrics)

        elif model_name in ("LSTM", "BiLSTM"):
            metrics = self._train_dl(model_name, pdict, X_tr, X_va, X_te,
                                  y_train, y_val, y_test, exp_id=exp_id)

        return metrics

    def _train_dl(self, model_name, params, X_tr, X_va, X_te,
                  y_train, y_val, y_test, exp_id=None):
        import tensorflow.keras.backend as K
        from src.modeling.deep_learning import LSTMModel, BiLSTMModel
        pdict = params if isinstance(params, dict) else {}
        seq = pdict.get("sequence_length", SEQUENCE_LENGTH)
        n_feat = X_tr.shape[1]

        X_tr_s, y_tr_s = create_sequences(X_tr, y_train, seq)
        X_va_s, y_va_s = create_sequences(X_va, y_val, seq)
        X_te_s, y_te_s = create_sequences(X_te, y_test, seq)

        if len(X_tr_s) == 0 or len(X_te_s) == 0:
            return None

        # Anchor residual: nilai aktual satu langkah sebelum tiap target
        # (sejajar dgn create_sequences: y_seq = y[seq:], anchor = y[seq-1:-1]).
        yt, yv, yte = (np.asarray(a, float) for a in (y_train, y_val, y_test))
        anc_tr = yt[seq - 1:-1]
        anc_va = yv[seq - 1:-1] if len(X_va_s) > 0 else None
        anc_te = yte[seq - 1:-1]

        dl_p = dict(sequence_length=seq, n_features=n_feat,
                    units=pdict.get("units", 64),
                    dropout=pdict.get("dropout", 0.2),
                    learning_rate=pdict.get("learning_rate", 0.001),
                    batch_size=pdict.get("batch_size", 16),
                    l1=pdict.get("l1", 0.0), l2=pdict.get("l2", 0.0),
                    epochs=pdict.get("epochs", 100))

        Cls = LSTMModel if model_name == "LSTM" else BiLSTMModel
        va_X = X_va_s if len(X_va_s) > 0 else None
        va_y = y_va_s if len(X_va_s) > 0 else None

        # ensemble multi-seed; catatan: metrik = rata-rata ensembel, hanya seed
        # pertama yang disimpan -> reproduksi penuh via evaluation_core
        preds = []
        preds_va = []
        first_model = None
        for seed in DL_ENSEMBLE_SEEDS:
            m = Cls(seed=seed, **dl_p)
            m.fit(X_tr_s, y_tr_s, va_X, va_y,
                  anchor_train=anc_tr, anchor_val=anc_va)
            preds.append(m.predict(X_te_s, anchor=anc_te))
            if va_X is not None and len(va_X) > 0:
                preds_va.append(m.predict(va_X, anchor=anc_va))
            if first_model is None:
                first_model = m
            else:
                del m
        pred_te = np.mean(preds, axis=0)
        metrics = evaluate_model(y_te_s, pred_te, model_name)
        # Catatan: segmen validasi dipakai sebagai set early stopping pelatihan
        # NN, jadi RMSE_val di sini juga bukan keluar-sampel murni.
        if preds_va:
            metrics = self._attach_val(metrics, y_va_s, np.mean(preds_va, axis=0))

        if metrics:
            metrics = self._save_model_helper(first_model, model_name, exp_id, metrics)

        del first_model
        K.clear_session()
        gc.collect()

        return metrics

    # Registry Update Helpers
    def _update_registry_from_focused(self):
        """Update registry with best result per model from focused evaluation."""
        if self.focused_results.empty:
            return

        grid_path = self.output_dir / "screening" / "experiment_grid.json"
        grid_dict = {}
        if grid_path.exists():
            with open(grid_path) as f:
                grid_dict = {c["id"]: c for c in json.load(f)}

        best_per_model = self.focused_results.loc[
            self.focused_results.groupby("model")["RMSE"].idxmin()
        ]
        updated_count = 0
        for _, row in best_per_model.iterrows():
            combo = grid_dict.get(row["experiment_id"], {})
            was_updated = update_if_better(self.best_registry, row["model"], {
                "experiment_id": row["experiment_id"],
                "RMSE": row["RMSE"],
                "MAPE": row.get("MAPE"),
                "R2": row.get("R2"),
                "source_phase": "focused",
                "combination": {
                    "imputation": row.get("imputation", combo.get("imputation")),
                    "features": row.get("features", combo.get("features")),
                    "scaler": row.get("scaler", combo.get("scaler")),
                    "feature_engineering": row.get("feature_engineering",
                                                   combo.get("feature_engineering")),
                },
                "label": row.get("label", combo.get("label", "")),
                "model_path": row.get("model_path"),
            })
            if was_updated:
                updated_count += 1
                rmse_val = row['RMSE']
                rmse_str = f"{rmse_val:.4f}" if pd.notna(rmse_val) else "N/A"
                print(f"    Registry updated: {row['model']} -> "
                      f"RMSE {rmse_str} ({row['experiment_id']})")

        save_registry(self.best_registry, self.registry_path)
        print(f"\n  Registry: {updated_count} model(s) diperbarui, "
              f"{len(self.best_registry)} total tercatat.")

    def _update_registry_from_tuning(self, tuning_df):
        """Update registry from tuning results — only if tuning improved over default."""
        if tuning_df.empty:
            return

        grid_path = self.output_dir / "screening" / "experiment_grid.json"
        grid_dict = {}
        if grid_path.exists():
            with open(grid_path) as f:
                grid_dict = {c["id"]: c for c in json.load(f)}

        updated_count = 0
        for _, row in tuning_df.iterrows():
            tuned_rmse = row.get("RMSE_tuned")
            default_rmse = row.get("RMSE_default")

            if pd.isna(tuned_rmse) or tuned_rmse == float("inf"):
                continue

            # Determine which RMSE to use — tuned or default
            use_tuned = (pd.notna(tuned_rmse) and
                         (pd.isna(default_rmse) or tuned_rmse <= default_rmse))
            best_rmse = tuned_rmse if use_tuned else default_rmse
            source = "tuning" if use_tuned else "focused"

            if not use_tuned:
                def_str = f"{default_rmse:.4f}" if pd.notna(default_rmse) else "N/A"
                tun_str = f"{tuned_rmse:.4f}" if pd.notna(tuned_rmse) else "N/A"
                print(f"    Tuning {row['model']} ({row['experiment_id']}): "
                      f"tidak membaik ({def_str} -> {tun_str}), "
                      f"registry tetap menggunakan hasil sebelumnya.")

            combo = grid_dict.get(row["experiment_id"], {})
            hp = None
            if use_tuned:
                try:
                    hp = json.loads(row.get("best_params", "{}").replace("'", '"'))
                except (json.JSONDecodeError, AttributeError):
                    hp = str(row.get("best_params"))

            focused_model_path = None
            if not use_tuned and not self.focused_results.empty:
                match_df = self.focused_results[
                    (self.focused_results["experiment_id"] == row["experiment_id"]) &
                    (self.focused_results["model"] == row["model"])
                ]
                if not match_df.empty:
                    focused_model_path = match_df["model_path"].values[0]

            was_updated = update_if_better(self.best_registry, row["model"], {
                "experiment_id": row["experiment_id"],
                "RMSE": best_rmse,
                "MAPE": row.get("MAPE_tuned") if use_tuned else None,
                "R2": row.get("R2_tuned") if use_tuned else None,
                "source_phase": source,
                "hyperparams": hp,
                "combination": {
                    "imputation": combo.get("imputation"),
                    "features": combo.get("features"),
                    "scaler": combo.get("scaler"),
                    "feature_engineering": combo.get("feature_engineering"),
                },
                "label": row.get("label", combo.get("label", "")),
                "model_path": row.get("model_path") if use_tuned else focused_model_path,
            })
            if was_updated:
                updated_count += 1

        save_registry(self.best_registry, self.registry_path)
        if updated_count > 0:
            print(f"\n  Registry: {updated_count} model(s) diperbarui dari tuning.")

    def _print_registry_comparison(self):
        """Print comparison between current run and historical best."""
        import math

        if not self.best_registry:
            return

        current_run = self.final_results[
            self.final_results["configuration"].str.contains("Run Saat Ini")
        ]
        if current_run.empty:
            return

        print(f"\n  PERBANDINGAN: Run Saat Ini vs Terbaik Historis")
        print(f"  {'Model':<12} {'Run Ini':<12} {'Historis':<12} "
              f"{'Sumber':<12} {'Status'}")
        print(f"  {'-'*60}")

        for _, row in current_run.iterrows():
            model = str(row["model"]) if row["model"] is not None else "?"
            current_rmse = row["RMSE"]
            hist = self.best_registry.get(model, {})
            hist_rmse = hist.get("RMSE")
            source = str(hist.get("source_phase", "?") or "?")

            # Guard: skip if either value is None
            if current_rmse is None or hist_rmse is None:
                print(f"  {model:<12} {'N/A':<12} {'N/A':<12} "
                      f"{source:<12} skip — nilai tidak tersedia")
                continue

            # Guard: skip if either value is not a valid number
            if not (isinstance(current_rmse, (int, float)) and
                    isinstance(hist_rmse, (int, float))):
                print(f"  {model:<12} {'N/A':<12} {'N/A':<12} "
                      f"{source:<12} N/A — tipe data tidak valid")
                continue

            # Guard: skip if inf or NaN
            if (math.isnan(current_rmse) or math.isnan(hist_rmse) or
                    math.isinf(current_rmse) or math.isinf(hist_rmse)):
                cur_str = f"{current_rmse:.4f}" if math.isfinite(current_rmse) else "inf/NaN"
                hist_str = f"{hist_rmse:.4f}" if math.isfinite(hist_rmse) else "inf/NaN"
                print(f"  {model:<12} {cur_str:<12} {hist_str:<12} "
                      f"{source:<12} N/A — nilai tidak valid (inf/NaN)")
                continue

            if current_rmse <= hist_rmse:
                status = "= terbaik" if current_rmse == hist_rmse else "RUN INI LEBIH BAIK"
            else:
                status = f"historis lebih baik (-{hist_rmse - current_rmse:+.4f})"

            print(f"  {model:<12} {current_rmse:<12.4f} {hist_rmse:<12.4f} "
                  f"{source:<12} {status}")

    # Load Helpers
    def _load_screening_results(self):
        path = self.output_dir / "screening" / "screening_results.csv"
        if path.exists():
            self.screening_results = pd.read_csv(path)
            print(f"  Loaded {len(self.screening_results)} screening results")

    def _load_focused_results(self):
        path = self.output_dir / "focused" / "focused_results.csv"
        if path.exists():
            self.focused_results = pd.read_csv(path)

    # Report Generators
    def _generate_screening_report(self, phase_dir, grid, ranking, elapsed):
        grid_dict = {c["id"]: c for c in grid}
        md = f"""# Laporan Fase 1: Hasil Screening

**Waktu Eksekusi:** {elapsed:.1f} detik
**Total Kombinasi Eksperimen:** {len(grid)}
**Model Evaluasi:** {', '.join(SCREENING_MODELS)}

Fase ini bertujuan untuk mengevaluasi secara cepat seluruh ruang kombinasi perlakuan data menggunakan algoritma yang efisien. Dari total {len(grid)} skenario, berikut adalah 20 kombinasi teratas yang mencatat error terendah.

## 20 Kombinasi Terbaik

| Peringkat | ID Skenario | Imputasi | Filter Fitur | Skala | Feature Engineering | Rata-rata RMSE |
|-----------|-------------|----------|--------------|-------|---------------------|----------------|
"""
        for rank, (_, row) in enumerate(ranking.head(20).iterrows(), 1):
            c = grid_dict.get(row["experiment_id"], {})
            md += (f"| {rank} | {row['experiment_id']} | "
                   f"{c.get('imputation', '?')} | {c.get('features', '?')} | "
                   f"{c.get('scaler', '?')} | {c.get('feature_engineering', '?')} | "
                   f"{row['avg_RMSE']:.4f} |\n")

        md += f"""
## Ringkasan Statistik

| Metrik | Nilai |
|--------|-------|
| Total skenario sukses | {len(ranking)} |
| RMSE rata-rata terbaik | {ranking['avg_RMSE'].min():.4f} |
| RMSE rata-rata terburuk | {ranking['avg_RMSE'].max():.4f} |
| Median performa (RMSE) | {ranking['avg_RMSE'].median():.4f} |

---
*Laporan dibuat pada: {datetime.now().strftime('%d %B %Y, %H:%M WIB')}*
"""
        with open(phase_dir / "screening_report.md", "w", encoding="utf-8") as f:
            f.write(md)

    def _generate_analysis_report(self, phase_dir, marginal, importance, interactions):
        imp_sorted = sorted(importance.items(), key=lambda x: x[1]["range"], reverse=True)
        md = f"""# Laporan Fase 2: Analisis Pengaruh Faktor

Tahap ini mengurai dampak spesifik tiap komponen perlakuan (imputasi, seleksi fitur, algoritma skala, dan rekayasa fitur) terhadap tingkat akurasi prediksi. Penilaian didasarkan pada selisih antara nilai konfigurasi terbaik dan terburuk pada setiap kategori.

## Tingkat Signifikansi Faktor (Diurutkan dari dampak paling tinggi)

| Peringkat | Komponen / Faktor | Rentang Dampak (RMSE) | Level Paling Optimal | Level Terburuk |
|-----------|------------------|----------------------|----------------------|----------------|
"""
        for rank, (f, info) in enumerate(imp_sorted, 1):
            md += (f"| {rank} | {f} | {info['range']:.4f} | "
                   f"{info['best_level']} | {info['worst_level']} |\n")

        md += "\n## Performa Rata-rata per Komponen\n\n"
        for factor, table in marginal.items():
            md += f"### Faktor Pilihan: {factor.title().replace('_', ' ')}\n\n"
            md += table.to_markdown() + "\n\n"

        md += "\n## Analisis Efek Interaksi (Pairwise)\n\n"
        for pair, pivot in interactions.items():
            md += f"### Relasi {pair.replace('_x_', ' terhadap ')}\n\n"
            md += pivot.to_markdown(floatfmt=".4f") + "\n\n"

        md += f"\n---\n*Laporan dibuat pada: {datetime.now().strftime('%d %B %Y, %H:%M WIB')}*\n"
        with open(phase_dir / "analysis_report.md", "w", encoding="utf-8") as f:
            f.write(md)

    def _generate_focused_report(self, phase_dir, elapsed):
        if self.focused_results.empty:
            return
        best = self.focused_results.loc[self.focused_results["RMSE"].idxmin()]
        md = f"""# Laporan Fase 3: Evaluasi Mendalam

Berbeda dengan tahapan *screening*, pada evaluasi mendalam seluruh kandidat dari kombinasi teratas diuji dengan jajaran model algoritma yang lengkap. Waktu eksekusi tahapan ini adalah {elapsed:.1f} detik.

## Pemenang Evaluasi (Konfigurasi Optimal Saat Ini)

Berikut adalah algoritma yang berhasil menorehkan angka error paling rendah:

| Parameter Utama | Kondisi Skenario Algoritma |
|----------------|----------------------------|
| ID Kombinasi   | {best['experiment_id']} — {best['label']} |
| Jenis Estimator| {best['model']} |
| Nilai RMSE     | {best['RMSE']:.4f} (Semakin kecil semakin kuat) |
| Simpangan MAPE | {best['MAPE']:.2f}% |
| Skor R²        | {best['R2']:.4f} |

## Data Perbandingan Selengkapnya

{self.focused_results.to_markdown(index=False, floatfmt='.4f')}

---
*Laporan dibuat pada: {datetime.now().strftime('%d %B %Y, %H:%M WIB')}*
"""
        with open(phase_dir / "focused_report.md", "w", encoding="utf-8") as f:
            f.write(md)

    def _generate_final_report(self, phase_dir):
        if self.final_results.empty:
            return

        # Build detail table for registry entries
        registry_detail = ""
        if self.best_registry:
            registry_detail = "\n## Detail Kombinasi Terbaik Keseluruhan (Registry)\n\n"
            registry_detail += (
                "Tabel berikut menunjukkan kombinasi preprocessing yang digunakan "
                "oleh masing-masing model pada hasil terbaik sepanjang seluruh eksperimen.\n\n"
            )
            registry_detail += (
                "| Model | ID | Imputasi | Seleksi Fitur | Scaler "
                "| Feature Engineering | RMSE | MAPE (%) | R2 | Sumber Fase |\n"
            )
            registry_detail += (
                "|-------|----|----------|---------------|--------"
                "|--------------------|------|----------|----|-------------|\n"
            )

            label_map = {
                "interpolate": "Interpolasi Linear", "ffill": "Forward Fill",
                "mean": "Mean Imputation", "knn": "KNN Imputation (k=5)",
                "all": "Semua 12 Fitur", "domain_itf": "Domain ITF (7)",
                "corr_top6": "Top-6 Korelasi", "mutual_info": "Mutual Info Top-7",
                "standard": "StandardScaler", "minmax": "MinMaxScaler",
                "robust": "RobustScaler",
                "none": "Tanpa FE", "lag": "Lag (t-1, t-3, t-6)",
                "rolling": "Rolling Mean/Std", "lag_rolling": "Lag + Rolling",
            }
            source_map = {
                "focused": "Fase 3", "tuning": "Fase 4 (Tuned)", "final": "Fase 5",
            }

            sorted_models = sorted(
                self.best_registry,
                key=lambda m: self.best_registry[m].get("RMSE", float("inf"))
            )
            for m in sorted_models:
                info = self.best_registry[m]
                combo = info.get("combination", {})
                imp = label_map.get(combo.get("imputation", "?"), combo.get("imputation", "?"))
                feat = label_map.get(combo.get("features", "?"), combo.get("features", "?"))
                sc = label_map.get(combo.get("scaler", "?"), combo.get("scaler", "?"))
                fe = label_map.get(combo.get("feature_engineering", "?"), combo.get("feature_engineering", "?"))
                src = source_map.get(info.get("source_phase", "?"), info.get("source_phase", "?"))
                rmse = info.get("RMSE", 0)
                mape = info.get("MAPE", 0)
                r2 = info.get("R2", 0)
                exp_id = info.get("experiment_id", "?")
                hp = info.get("hyperparams")

                registry_detail += (
                    f"| {m} | {exp_id} | {imp} | {feat} | {sc} "
                    f"| {fe} | {rmse:.4f} | {mape:.2f} | {r2:.4f} | {src} |\n"
                )

            # Add hyperparameter detail if any model was tuned
            tuned_models = [m for m in sorted_models
                            if self.best_registry[m].get("hyperparams")]
            if tuned_models:
                registry_detail += "\n### Hyperparameter Hasil Tuning\n\n"
                for m in tuned_models:
                    hp = self.best_registry[m].get("hyperparams")
                    registry_detail += f"- **{m}**: `{hp}`\n"
                registry_detail += "\n"

        md = f"""# Laporan Fase 5: Evaluasi Komparatif Komprehensif

Dalam fase puncak ini, kita mengkomparasi efisiensi dari penalaan struktur algoritma terpilih dengan model parameter acuan (baseline) klasik. Hal ini memastikan setiap modifikasi eksperimen telah meningkatkan prediksi data *Time Series*.

## Ringkasan Banding Data

Tabel di bawah menggabungkan setiap arsitektur model dan seberapa jauh peningkatan akurasinya:

{self.final_results.to_markdown(index=False, floatfmt='.4f')}
{registry_detail}
---
*Laporan dibuat pada: {datetime.now().strftime('%d %B %Y, %H:%M WIB')}*
"""
        with open(phase_dir / "final_report.md", "w", encoding="utf-8") as f:
            f.write(md)
