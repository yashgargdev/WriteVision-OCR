# WriteVision — ML Project Showcase (Small PPT)

This is a condensed, small-deck version (8 slides) for quick presentation.

---

## Slide 1 — Title
- WriteVision: Handwritten OCR (Local Fine-Tune)
- Presenter: Yash [Your Full Name]
- College: [Your College Name], Date: [Presentation Date]

---

## Slide 2 — Problem & Goal
- Problem: Recognize handwritten/stylized text from images.
- Constraints: small dataset, varied handwriting, noisy backgrounds.
- Goal: Build a local OCR that fine-tunes on custom data.

---

## Slide 3 — Solution Overview
- Seq2Seq model: vision encoder + text decoder.
- Fine-tuned locally on labeled images.
- Simple web UI via Flask at `http://127.0.0.1:5000/`.

---

## Slide 4 — Data & Training
- Data: `data/train/images` + labels in `data/train/labels.csv` (24 samples).
- Strategy: Freeze encoder; train decoder-only for stability.
- Settings: 30 epochs, batch size 2; checkpoints to `models\writevision-trocr-finetuned`.
- Decoding controls: `no_repeat_ngram_size=2`, `repetition_penalty=1.2`.

---

## Slide 5 — Results & Demo
- Training samples recognized consistently: `Second`, `Parbhat`, `BraWl StAr`.
- Loss to ~0.0001–0.0005 by epoch 30.
- Demo: Open the app and upload sample images to view predictions.

---

## Slide 6 — Updates Since Review 1
- Before: CLI-only, unstable outputs.
- After: Web UI added; encoder frozen; repetition controls; more data (7 → 24); 30-epoch training; fixed loading/script issues.

---

## Slide 7 — Challenges & Lessons
- Small data → limited generalization.
- Consistent preprocessing matters.
- Decoder-only fine-tuning improves stability on tiny datasets.

---

## Slide 8 — Next Steps & Q&A
- Collect more diverse data; add validation metrics (CER/WER).
- Consider greedy decoding for determinism.
- Try light augmentations; package as a desktop/mobile demo.
- Q&A.