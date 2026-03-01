import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xgboost as xgb
import joblib
import os


# ==========================================
# ✅ 1️⃣  CNN Model
# ==========================================
class ECG1DCNN(nn.Module):
    def __init__(self, num_classes):
        super(ECG1DCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.25)

        self.flatten_dim = 128 * 81

        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ==========================================
# ✅ 2️⃣  XGBoost Wrapper
# ==========================================
class ECGXGBoostWrapper:
    """
    Wrapper so XGBoost model behaves similarly to PyTorch model for inference
    """

    def __init__(self, model_path=None, label_encoder_path=None):
        self.model = None
        if model_path:
            self.load_model(model_path)

        self.label_encoder = None
        if label_encoder_path:
            self.label_encoder = joblib.load(label_encoder_path)

    def load_model(self, model_path):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ XGBoost model not found: {model_path}")

        # XGBoost JSON/UBJ Model
        if model_path.endswith(".json") or model_path.endswith(".ubj"):
            self.model = xgb.Booster()
            self.model.load_model(model_path)

        else:
            raise ValueError("❌ Unsupported XGBoost format. Use .json or .ubj")

    def predict(self, input_data):
        """
        input_data: 1D array of ECG floats
        return: class_id, probability_list
        """

        x = np.array(input_data).reshape(1, -1)
        dtest = xgb.DMatrix(x)

        preds = self.model.predict(dtest)

        # Multi-class
        if preds.ndim > 1:
            class_id = int(np.argmax(preds[0]))
            return class_id, preds[0].tolist()

        # Single class output
        class_id = int(preds[0])
        return class_id, None



import numpy as np
import joblib
import os

class AdaBoostECGWrapper:
    """
    Wrapper class to make AdaBoost behave similar to PyTorch/XGBoost inference interface.
    """

    def __init__(self, model_path=None, label_encoder_path=None):
        self.model = None

        if model_path:
            self.load_model(model_path)

        self.label_encoder = None
        if label_encoder_path:
            self.label_encoder = joblib.load(label_encoder_path)

    def load_model(self, model_path):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ AdaBoost model not found: {model_path}")

        if model_path.endswith(".pkl"):
            self.model = joblib.load(model_path)
        else:
            raise ValueError("❌ Unsupported AdaBoost format. Use .pkl")

    def predict(self, input_data):
        """
        input_data: 1D ECG float list
        returns: class_id, probabilities
        """

        x = np.array(input_data).reshape(1, -1)

        preds = self.model.predict(x)
        class_id = int(preds[0])

        # Probability (may be None if estimator does not support predict_proba)
        try:
            probs = self.model.predict_proba(x)[0].tolist()
        except:
            probs = None

        return class_id, probs
