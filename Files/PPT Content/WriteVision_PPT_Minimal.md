# WriteVision — Minimal Presentation Deck

Slide 1 — Title
- WriteVision: Handwritten Text Recognition (OCR)
- Presenter: Yash • College • Date

Slide 2 — Technology Used
- Python with `Flask`, `PyTorch`, `transformers`
- Hugging Face TrOCR as the base OCR model
- Local environment (Windows); training logs via `tqdm`
- Simple web UI served from `static/index.html`

Slide 3 — Model Used
- Encoder–decoder OCR (TrOCR architecture)
- Fine-tuned locally: encoder frozen, decoder trained
- Model saved at `models/writevision-trocr-finetuned`

Slide 4 — Dataset
- Custom dataset at `data/train/` (`images/` + `labels.csv`)
- 24 labeled handwritten samples (e.g., "Second", "Parbhat", "Brawl Star")
- Preprocessing with TrOCR processor (resize/normalize)

Slide 5 — How I Trained the Model
- Batch size: `2` • Epochs: `30` • Checkpoint saved each epoch
- Loss improved from `4.59` (Epoch 1) to `<0.001` (Epoch 15+)
- Decoder-only fine-tuning for stability and speed
- Inference uses tuned decoding (greedy/beam as needed)

Slide 6 — Output of the Program
- Consistent predictions on project samples:
  - "Second" → "Second"
  - "Parbhat" → "Parbhat"
  - "Brawl Star" → "Brawl Star"
- Web UI: upload image → get text at `http://127.0.0.1:5000/`

Slide 7 — How Many Times Trained
- Retrained once post-Review 1 for `30` epochs
- Checkpoints saved per epoch for comparison/rollback

Slide 8 — Changes Since Review 1
- Added a simple web UI (Flask + HTML)
- Refactored training pipeline; decoder-only fine-tuning
- Expanded dataset using my own images; enforced label consistency
- Fixed incorrect outputs via decoding and preprocessing adjustments
- Deployed local fine-tuned model: `writevision-trocr-finetuned`