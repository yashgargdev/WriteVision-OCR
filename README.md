# WriteVision OCR

WriteVision is a lightweight, local handwritten text recognition (OCR) project. It fine-tunes a Hugging Face TrOCR encoder–decoder model on a small custom dataset and serves predictions through a simple Flask web UI and REST API.

## Features
- Local fine-tuning of TrOCR with decoder-only training for stability on tiny datasets
- Simple web UI at `http://127.0.0.1:5000/` for drag-and-drop image recognition
- REST API for programmatic recognition: `POST /api/recognize`
- Optional text export endpoint: `POST /api/export_txt`
- Checkpoints saved after each epoch for reproducibility

## Tech Stack
- `Python`, `Flask`, `PyTorch`, `transformers`
- Base model: `microsoft/trocr-base-handwritten` (or local fine-tuned model)
- Runs on Windows (PowerShell/CMD) and CPU by default; CUDA if available

## Project Structure
```
WriteVision/
├── app.py                       # Flask app (UI + API)
├── train.py                     # Fine-tuning script
├── run_writevision.cmd          # Convenience script: venv, deps, train, run
├── static/index.html            # Web UI
├── data/train/labels.csv        # Labels (filename,text or filename\ttext)
├── models/writevision-trocr-finetuned/  # Fine-tuned model dir (configs)
├── requirements.txt             # Python dependencies
├── Files/PPT Content/           # Presentation/report materials
└── .gitignore                   # Excludes venv, caches, raw images, large weights
```

## Getting Started
1) Clone the repo
```
 git clone https://github.com/yashgargdev/WriteVision-OCR.git
 cd WriteVision-OCR
```

2) Use the helper script (recommended on Windows)
```
 run_writevision.cmd
```
- Creates/activates `.venv`
- Installs dependencies from `requirements.txt` if missing
- Uses local fine-tuned model if available, otherwise base TrOCR
- Starts the server on `http://127.0.0.1:5000/`

Manual setup (alternative):
```
 python -m venv .venv
 .venv\Scripts\activate
 pip install -r requirements.txt
 # Optional: set the model to your fine-tuned path
 set WRITEVISION_MODEL=models\writevision-trocr-finetuned
 python app.py
```

## Web UI
- Open `http://127.0.0.1:5000/`
- Upload an image and view the recognized text

## REST API
Recognize text from an image:
```
POST /api/recognize
Form-Data: image=@path/to/image.jpg
Response: { "text": "..." }
```
Example with `curl`:
```
curl -X POST -F image=@data/train/images/1.jpg http://127.0.0.1:5000/api/recognize
```

Export text to a downloadable `.txt` file:
```
POST /api/export_txt
JSON: { "text": "Your recognized text" }
Response: attachment (text/plain)
```

## Training
Use the helper script with the `train` label to avoid Windows parsing issues:
```
 run_writevision.cmd train <epochs> <batch_size> <save_dir>
```
Examples:
- Train for 30 epochs, batch size 2, save to the default model dir
```
 run_writevision.cmd train 30 2 models\writevision-trocr-finetuned
```
- Defaults if omitted: `epochs=3`, `batch_size=2`, `save_dir=models\writevision-trocr-finetuned`

Direct Python invocation:
```
 python train.py \
   --model_name %WRITEVISION_MODEL% \
   --image_root data/train/images \
   --labels_csv data/train/labels.csv \
   --save_dir models/writevision-trocr-finetuned \
   --epochs 30 --batch_size 2 --lr 5e-5 --warmup_ratio 0.1 --max_target_length 32
```

Training notes:
- Encoder is frozen; only the decoder is trained to minimize feature drift on small data
- Decoding constraints during eval: `num_beams=5`, `no_repeat_ngram_size=2`, `repetition_penalty=1.2`
- Checkpoints saved each epoch to `save_dir`

## Dataset Format
- Place images in `data/train/images/`
- Labels in `data/train/labels.csv` as either `filename,text` or `filename\ttext`
- Example rows:
```
1.jpg,Brawl Star
2.jpg,Parbhat
3.jpg,Second
```
Tips:
- Keep casing, spacing, and punctuation consistent—the model learns labels exactly as written
- The repo `.gitignore` excludes raw images by default to keep the repo small

## Model Selection
- The app reads `WRITEVISION_MODEL`
  - If not set, uses `microsoft/trocr-base-handwritten`
  - If your fine-tuned model exists at `models\writevision-trocr-finetuned`, the helper script sets it automatically

## Results (Current Project)
- Trained on 24 custom samples for 30 epochs
- Stable predictions for: `Second`, `Parbhat`, `Brawl Star`
- Loss decreased from ~`4.59` (early) to `~0.0001–0.0005` (late)

## Changes Since Review 1
- Added Flask web UI and REST endpoints
- Refactored training pipeline; decoder-only fine-tuning
- Expanded dataset (7 → 24) with consistent labeling
- Improved decoding and preprocessing; fixed prior incorrect outputs

## Acknowledgements
- Hugging Face Transformers and TrOCR
- PIL/Pillow for image handling
- Flask for the web server

## License
- Personal/educational use. Add a license if you plan to distribute.