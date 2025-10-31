import io
import os
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from PIL import Image
from PIL import ImageOps

# Hugging Face Transformers for TrOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

app = Flask(__name__)

# Lazy global cache for model/processor
MODEL_CACHE = {
    "processor": None,
    "model": None,
    "device": None,
}

MODEL_NAME = os.environ.get("WRITEVISION_MODEL", "microsoft/trocr-base-handwritten")


def get_model_and_processor():
    """Load TrOCR model and processor lazily and cache on first use."""
    if MODEL_CACHE["model"] is not None:
        return MODEL_CACHE["processor"], MODEL_CACHE["model"], MODEL_CACHE["device"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    MODEL_CACHE["processor"] = processor
    MODEL_CACHE["model"] = model
    MODEL_CACHE["device"] = device
    return processor, model, device


# Add a simple image preprocessing step to improve contrast for handwriting
def preprocess_image(img: Image.Image) -> Image.Image:
    # Convert to grayscale, apply autocontrast, then back to RGB
    try:
        g = img.convert("L")
        g = ImageOps.autocontrast(g)
        return g.convert("RGB")
    except Exception:
        # Fallback: return original image if preprocessing fails
        return img


@app.route("/api/recognize", methods=["POST"])
def recognize():
    """Recognize handwritten text from an image upload."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    # Preprocess image to improve OCR signal
    image = preprocess_image(image)

    processor, model, device = get_model_and_processor()

    # Preprocess image for TrOCR
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        # Use beam search for better accuracy, avoid sampling
        generated_ids = model.generate(
            pixel_values,
            num_beams=5,
            early_stopping=True,
            max_length=32,
            do_sample=False,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
        )
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return jsonify({"text": text})


@app.route("/api/export_txt", methods=["POST"])
def export_txt():
    """Export provided text as a downloadable .txt file."""
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not isinstance(text, str):
        return jsonify({"error": "Invalid text"}), 400

    fname = f"recognized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    bytes_io = io.BytesIO(text.encode("utf-8"))
    bytes_io.seek(0)
    return send_file(
        bytes_io,
        mimetype="text/plain",
        as_attachment=True,
        download_name=fname,
    )


@app.route("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    # Important for local dev on Windows
    app.static_folder = "static"
    app.run(host="127.0.0.1", port=5000, debug=True)