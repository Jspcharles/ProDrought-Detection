from tensorflow.keras.utils import plot_model
from cnn_lstm_models import build_model
import os

output_dir = "models/architectures"
os.makedirs(output_dir, exist_ok=True)

model_types = ["A1", "A2", "A4", "A6"]

for model_type in model_types:
    model = build_model(model_type)
    plot_model(
        model,
        to_file=os.path.join(output_dir, f"model_{model_type}.png"),
        show_shapes=True,
        show_layer_names=True
    )
