# Random Forest and XGBoost model implementation

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib


class RandomForestModel:
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(
            n_jobs=-1,
            **kwargs,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("Random Forest fitted")

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)


class XGBoostModel:
    def __init__(self, **kwargs):
        kwargs.setdefault("verbosity", 0)
        # early_stopping_rounds dipakai hanya saat ada validation set
        self.early_stopping_rounds = kwargs.pop("early_stopping_rounds", 20)
        self.model = XGBRegressor(
            **kwargs,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None and len(X_val) > 0:
            self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)
        print("XGBoost fitted")

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        if filepath.endswith(".pkl"):
            import joblib
            joblib.dump(self.model, filepath)
        else:
            self.model.save_model(filepath)

    def load(self, filepath):
        if filepath.endswith(".pkl"):
            import joblib
            self.model = joblib.load(filepath)
        else:
            self.model.load_model(filepath)