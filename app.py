import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from model import PlantDiseaseModel

MODEL_PATH = Path(__file__).resolve().with_name("efficientnet_b0_plant_disease.pth")
LABELS_PATH = MODEL_PATH.with_name("class_names.json")
DISPLAY_LABELS_PATH = MODEL_PATH.with_name("class_names_display.json")

predictor = PlantDiseaseModel(MODEL_PATH, labels_path=LABELS_PATH)

app = Flask(__name__)


def load_display_class_names() -> list[str] | None:
    if not DISPLAY_LABELS_PATH.exists():
        return None

    try:
        data = json.loads(DISPLAY_LABELS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if isinstance(data, list):
        return [str(item) for item in data]
    return None


@app.route("/")
def index():
    return render_template(
        "index.html",
        class_names=predictor.get_class_names(),
        class_names_display=load_display_class_names(),
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing image field"}), 400

    file_storage = request.files["image"]
    if file_storage.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        result = predictor.predict(file_storage.stream)
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
