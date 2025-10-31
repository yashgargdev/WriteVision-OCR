import os
import csv
import math
import random
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import AdamW, get_linear_schedule_with_warmup

# ---------------------------
# Config & Utilities
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_image(img: Image.Image) -> Image.Image:
    """Match inference-time preprocessing to help training stability."""
    try:
        g = img.convert("L")
        g = ImageOps.autocontrast(g)
        return g.convert("RGB")
    except Exception:
        return img


def read_labels_csv(csv_path: str) -> List[Dict[str, str]]:
    """Read labels.csv that maps filename to transcription.
    Supports comma or tab-separated values.
    Returns a list of dicts: {"filename": str, "text": str}
    """
    records: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        raw = f.read().strip().splitlines()
    for line in raw:
        if not line.strip():
            continue
        # Try tab, then comma
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split(",")
        if len(parts) < 2:
            # Skip malformed
            continue
        fname = parts[0].strip()
        text = parts[1].strip()
        if fname and text:
            records.append({"filename": fname, "text": text})
    return records


# ---------------------------
# Dataset
# ---------------------------
class HandwritingDataset(Dataset):
    def __init__(self, image_root: str, labels: List[Dict[str, str]], processor: TrOCRProcessor, max_target_length: int = 32):
        self.image_root = image_root
        self.labels = labels
        self.processor = processor
        self.max_target_length = max_target_length

        # Filter out missing files
        existing = []
        for rec in self.labels:
            fp = os.path.join(self.image_root, rec["filename"])
            if os.path.isfile(fp):
                existing.append(rec)
        missing = len(self.labels) - len(existing)
        if missing:
            print(f"[train.py] Warning: {missing} label entries refer to missing image files and were skipped.")
        self.labels = existing

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rec = self.labels[idx]
        img_path = os.path.join(self.image_root, rec["filename"])
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        im = preprocess_image(im)

        # Processor returns pixel_values shape (1, C, H, W)
        pixel_values = self.processor(images=im, return_tensors="pt").pixel_values[0]

        # Tokenize label text
        text = rec["text"]
        labels = self.processor.tokenizer(
            text,
            truncation=True,
            max_length=self.max_target_length,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch, processor: TrOCRProcessor):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]
    # Pad labels to max length in batch
    labels_batch = processor.tokenizer.pad(
        {"input_ids": labels},
        padding=True,
        return_tensors="pt",
    )
    labels = labels_batch["input_ids"]
    # Replace pad token ids with -100 so they are ignored by the loss
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------
# Training Loop
# ---------------------------

def train(
    model_name: str,
    image_root: str,
    labels_csv: str,
    save_dir: str = "models/writevision-trocr-finetuned",
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 5e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.0,
    max_target_length: int = 32,
    seed: int = 42,
):
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train.py] Using device: {device}")

    print(f"[train.py] Loading processor & model: {model_name}")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    # Configure decoder start/pad/eos ids to avoid training errors
    tokenizer = processor.tokenizer
    # Use CLS as start token and SEP as EOS for RoBERTa-like decoders
    if hasattr(tokenizer, "cls_token_id") and tokenizer.cls_token_id is not None:
        model.config.decoder_start_token_id = tokenizer.cls_token_id
    elif hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id
    else:
        raise ValueError("Tokenizer missing cls/bos token id; cannot set decoder_start_token_id.")
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(tokenizer, "sep_token_id") and tokenizer.sep_token_id is not None:
        model.config.eos_token_id = tokenizer.sep_token_id
    elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    # Ensure vocab size is aligned
    model.config.vocab_size = len(tokenizer)
    model.to(device)

    # Freeze encoder for tiny datasets to stabilize training and improve memorization
    for p in model.encoder.parameters():
        p.requires_grad = False
    print("[train.py] Encoder frozen; training decoder only.")

    # Load dataset
    records = read_labels_csv(labels_csv)
    if not records:
        raise RuntimeError(f"No valid records found in {labels_csv}. Ensure it contains 'filename, text' or 'filename\ttext'.")

    dataset = HandwritingDataset(image_root, records, processor, max_target_length=max_target_length)
    if len(dataset) == 0:
        raise RuntimeError("Dataset has zero valid images. Check paths and filenames.")

    # DataLoader
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, processor))

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    total_steps = math.ceil(len(dl) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"[train.py] Training for {epochs} epochs on {len(dataset)} samples; batch_size={batch_size}, total_steps={total_steps}")

    model.train()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch in dl:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            running_loss += loss.item()

            if global_step % 10 == 0:
                avg_loss = running_loss / 10
                print(f"[train.py] Epoch {epoch} | step {global_step} | loss {avg_loss:.4f}")
                running_loss = 0.0

        # End of epoch: simple qualitative check on a few samples
        model.eval()
        with torch.no_grad():
            sample_count = min(3, len(dataset))
            print(f"[train.py] Sample predictions after epoch {epoch}:")
            for i in range(sample_count):
                sample = dataset[i]
                pv = sample["pixel_values"].unsqueeze(0).to(device)
                gen_ids = model.generate(
                    pv,
                    num_beams=5,
                    early_stopping=True,
                    max_length=max_target_length,
                    do_sample=False,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.2,
                )
                pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                print(f"  - {dataset.labels[i]['filename']}: '{pred}' (target: '{dataset.labels[i]['text']}')")
        model.train()

        # Save checkpoint each epoch
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        print(f"[train.py] Saved checkpoint to: {save_dir}")

    print("[train.py] Training complete.")
    print("[train.py] To use this model in the app, set the environment variable WRITEVISION_MODEL to the saved directory.")
    print(f"  Example (PowerShell): $env:WRITEVISION_MODEL = '{os.path.abspath(save_dir)}'")


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on your handwriting dataset.")
    parser.add_argument("--model_name", type=str, default=os.environ.get("WRITEVISION_MODEL", "microsoft/trocr-base-handwritten"), help="Base model to fine-tune (HF hub id or local path)")
    parser.add_argument("--image_root", type=str, default=os.path.join("data", "train", "images"), help="Directory containing training images")
    parser.add_argument("--labels_csv", type=str, default=os.path.join("data", "train", "labels.csv"), help="CSV mapping filename to text (tab or comma-separated)")
    parser.add_argument("--save_dir", type=str, default=os.path.join("models", "writevision-trocr-finetuned"), help="Directory to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup steps ratio")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--max_target_length", type=int, default=32, help="Max target length for labels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        image_root=args.image_root,
        labels_csv=args.labels_csv,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_target_length=args.max_target_length,
        seed=args.seed,
    )