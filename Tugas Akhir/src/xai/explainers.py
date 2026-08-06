import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SHAPExplainer:
    def __init__(self, model, model_type="tree"):
        """
        model: the raw sklearn/xgboost model (not the wrapper class).
        model_type: 'tree' for tree-based, 'kernel' for anything else.
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None

    def fit(self, X_background):
        if self.model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict, shap.sample(X_background, 100)
            )

    def explain(self, X):
        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values

    def get_global_importance(self, feature_names):
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        return df.sort_values("mean_abs_shap", ascending=False)

    def explain_instance(self, X_instance, feature_names):
        vals = self.explainer.shap_values(X_instance.reshape(1, -1))[0]
        result = dict(zip(feature_names, vals))
        return dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True))

    def plot_summary(self, X, feature_names, save_path="results/shap_summary.png"):
        if self.shap_values is None:
            self.explain(X)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, feature_names=feature_names, show=False)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary plot saved to {save_path}")

    def plot_waterfall(self, X_instance, feature_names, save_path="results/shap_waterfall.png"):
        vals = self.explainer.shap_values(X_instance.reshape(1, -1))[0]
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=vals,
                base_values=self.explainer.expected_value,
                data=X_instance,
                feature_names=feature_names,
            ),
            show=False,
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"SHAP waterfall plot saved to {save_path}")


class LIMEExplainer:
    def __init__(self, X_train, feature_names):
        """
        X_train: training data array for distribution estimation.
        feature_names: list of feature names.
        """
        self.feature_names = feature_names
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            mode="regression",
            verbose=False,
            random_state=42,  # agar hasil reprodusibel
        )

    def explain_instance(self, predict_fn, X_instance, num_features=10):
        explanation = self.explainer.explain_instance(
            X_instance, predict_fn, num_features=num_features
        )
        return {
            "contributions": dict(explanation.as_list()),
            "local_prediction": explanation.local_pred[0],
            "r2_score": explanation.score,
        }

    def plot_explanation(self, predict_fn, X_instance, save_path="results/lime_explanation.png"):
        explanation = self.explainer.explain_instance(
            X_instance, predict_fn, num_features=10
        )
        fig = explanation.as_pyplot_figure()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"LIME explanation plot saved to {save_path}")


def format_xai_output(shap_dict, lime_dict, prediction):
    """Format SHAP + LIME output into a single dict for LLM or JSON export."""
    sorted_shap = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
    top_pos = [(k, v) for k, v in sorted_shap if v > 0][:5]
    top_neg = [(k, v) for k, v in sorted_shap if v < 0][-5:]

    return {
        "prediction": float(prediction),
        "shap_analysis": {
            "top_positive_contributors": top_pos,
            "top_negative_contributors": top_neg,
            "all_contributions": shap_dict,
        },
        "lime_analysis": {
            "local_prediction": lime_dict["local_prediction"],
            "contributions": lime_dict["contributions"],
            "model_fit_r2": lime_dict["r2_score"],
        },
    }