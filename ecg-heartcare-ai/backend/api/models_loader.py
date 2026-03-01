from .model_architecture import ECG1DCNN, ECGXGBoostWrapper, AdaBoostECGWrapper

MODEL_MAP = {
    "ECG1DCNN": {
        "backend": "torch",
        "class": ECG1DCNN,
        "path": "api/models/best_ecg_1dcnn.pth",
        "num_classes": 4,
        "input_size": 2604,
    },

    "ECGXGBoost": {
        "backend": "xgboost",
        "class": ECGXGBoostWrapper,
        "path": "api/models/best_xgb_ecg.json",
        "num_classes": 4,
        "input_size": 2604,
    },

    "ECGAdaBoost": {
        "backend": "adaboost",
        "class": AdaBoostECGWrapper,
        "path": "api/models/best_adaboost_ecg.pkl",
        "num_classes": 4,
        "input_size": 2604,
    }
}
