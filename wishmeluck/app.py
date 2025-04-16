from flask import Flask, request, render_template
import torch
import numpy as np
import json
from model.minimal_model import GCN_TCN_CapsNet

# Mapping from class index to activity name
LABEL_TO_ACTIVITY = {
    0: "Walking",
    1: "Jogging",
    2: "Sitting",
    3: "Standing",
    4: "Upstairs",
    5: "Downstairs"
}

app = Flask(__name__)

# Predict using sliding window with step control
def predict_with_window_stride(sample, stride=30):
    device = torch.device("cpu")

    # Load the model
    model = GCN_TCN_CapsNet(input_dim=3, num_classes=6, num_nodes=128)
    model.load_state_dict(torch.load("model/saved_model.pth", map_location=device))
    model.to(device)
    model.eval()

    results = []

    if sample.shape[0] < 128:
        padding = np.zeros((128 - sample.shape[0], sample.shape[1]))
        sample = np.vstack((sample, padding))

    # Sliding window with stride
    for start_idx in range(0, sample.shape[0] - 128 + 1, stride):
        window = sample[start_idx:start_idx + 128]
        window = window[:, :3]  # Keep only accelerometer data

        input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_tensor)
            predicted_class = torch.argmax(pred, dim=1).item()
            activity = LABEL_TO_ACTIVITY.get(predicted_class, "Unknown")
            results.append((predicted_class, activity))

    return results

# Flask route
@app.route("/", methods=["GET", "POST"])
def index():
    last_prediction = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            sample = np.array(json.load(file), dtype=np.float32)
            predictions = predict_with_window_stride(sample, stride=30)
            if predictions:
                last_prediction = predictions[-1]  # Show the last prediction from all windows

    return render_template("index.html",
                           class_idx=last_prediction[0] if last_prediction else None,
                           activity=last_prediction[1] if last_prediction else None)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
