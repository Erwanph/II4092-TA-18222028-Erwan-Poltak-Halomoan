"""Registry JSON hasil terbaik per model, persisten antar-run pipeline."""

import json
from pathlib import Path
from datetime import datetime


REGISTRY_FILENAME = "best_results_registry.json"


def load_registry(registry_path):
    """Load the best results registry from disk, or return empty dict."""
    path = Path(registry_path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(registry, registry_path):
    """Save the registry to disk."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False, default=str)


def update_if_better(registry, model, result):
    new_rmse = result.get("RMSE")

    if new_rmse is None or new_rmse == float("inf"):
        return False

    current = registry.get(model)

    # Entri kanonis yang dikunci tak boleh ditimpa run pipeline (angka buku).
    if current and current.get("locked"):
        print(f"  [REGISTRY] {model} terkunci; hasil baru diabaikan "
              f"(RMSE {new_rmse:.4f}).")
        return False

    if current is None:
        should_update = True
    else:
        current_rmse = current.get("RMSE")

        if current_rmse is None:
            should_update = True
        else:
            should_update = new_rmse < current_rmse

    if should_update:
        new_path = result.get("model_path")
        # Hapus berkas model lama yang tergantikan agar models/ hanya berisi terbaik
        old_path = (current or {}).get("model_path")
        if old_path and old_path != new_path:
            try:
                Path(old_path).unlink()
            except OSError:
                pass

        registry[model] = {
            "experiment_id": result.get("experiment_id", "?"),
            "RMSE": new_rmse,
            "MAPE": result.get("MAPE"),
            "R2": result.get("R2"),
            "source_phase": result.get("source_phase", "unknown"),
            "hyperparams": result.get("hyperparams"),
            "combination": result.get("combination", {}),
            "label": result.get("label", ""),
            "model_path": new_path,
            "updated_at": datetime.now().isoformat(),
        }
        return True

    # Kandidat kalah: buang berkasnya jika ada (bukan model terbaik)
    losing_path = result.get("model_path")
    keep_paths = {info.get("model_path") for info in registry.values()}
    if losing_path and losing_path not in keep_paths:
        try:
            Path(losing_path).unlink()
        except OSError:
            pass

    return False

def prune_model_files(registry, models_dir="models", keep_prefixes=("expert_eval",)):
    """Sisakan hanya berkas model yang dirujuk registry (model terbaik per algoritma).

    Berkas .pkl/.keras lain di models/ dihapus, kecuali yang berawalan keep_prefixes
    (mis. bundel evaluasi). Kembalikan jumlah berkas yang dihapus.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return 0
    keep = set()
    for info in registry.values():
        mp = info.get("model_path")
        if mp:
            keep.add(Path(mp).resolve())

    removed = 0
    for f in models_dir.iterdir():
        if f.suffix not in (".pkl", ".keras"):
            continue
        if any(f.name.startswith(p) for p in keep_prefixes):
            continue
        if f.resolve() not in keep:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def get_best_for_model(registry, model):
    """Get the best historical result for a specific model."""
    return registry.get(model)
