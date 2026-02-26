"""
Train a skin condition classifier from images (e.g. skin disease dataset).
Supports: (1) folder structure images_dir/<condition>/*.jpg  (2) CSV with image_name, condition.
Used to recommend SkinMe products by predicted condition.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from skin_assistant.config import get_settings


def _collect_image_labels_from_folders(images_dir: Path) -> Tuple[List[str], List[str]]:
    """Scan images_dir for subdirs; each subdir name = condition, files = images. Returns (paths, labels)."""
    paths, labels = [], []
    if not images_dir.exists():
        return paths, labels
    for subdir in sorted(images_dir.iterdir()):
        if not subdir.is_dir():
            continue
        condition = subdir.name
        for f in subdir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                paths.append(str(f))
                labels.append(condition)
    return paths, labels


def _collect_image_labels_from_csv(
    csv_path: Path, images_dir: Path, image_col: str = "image_name", condition_col: str = "condition"
) -> Tuple[List[str], List[str]]:
    """CSV with image_col and condition_col; images live in images_dir (filename = image_name or path)."""
    paths, labels = [], []
    if not csv_path.exists() or not images_dir.exists():
        return paths, labels
    df = pd.read_csv(csv_path)
    if image_col not in df.columns or condition_col not in df.columns:
        return paths, labels
    for _, row in df.iterrows():
        name = str(row[image_col]).strip()
        cond = str(row[condition_col]).strip()
        if not name or not cond:
            continue
        # Support full path or just filename
        p = images_dir / name
        if not p.exists():
            p = images_dir / (name if not Path(name).suffix else name + ".jpg")
        if p.exists():
            paths.append(str(p))
            labels.append(cond)
    return paths, labels


def train_skin_condition_classifier(
    images_dir: Optional[Path] = None,
    labels_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    image_col: str = "image_name",
    condition_col: str = "condition",
    epochs: int = 5,
    batch_size: int = 16,
    image_size: int = 224,
) -> dict:
    """
    Train image classifier for skin conditions. Saves model to output_dir/skin_condition_model.pt.
    If labels_csv is set, uses CSV (image_name, condition); else uses folder structure images_dir/<condition>/*.jpg.
    """
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms, models
        from PIL import Image
    except ImportError:
        return {"error": "Install torch, torchvision, Pillow: pip install torch torchvision Pillow"}

    settings = get_settings()
    images_dir = images_dir or settings.skin_disease_images_dir
    output_dir = output_dir or settings.models_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if labels_csv and Path(labels_csv).exists():
        image_paths, image_labels = _collect_image_labels_from_csv(
            Path(labels_csv), images_dir, image_col=image_col, condition_col=condition_col
        )
    else:
        image_paths, image_labels = _collect_image_labels_from_folders(images_dir)

    if len(image_paths) < 2 or len(set(image_labels)) < 2:
        return {
            "error": "Need at least 2 images and 2 classes. Use folder structure data/skin_disease_images/<condition>/*.jpg or CSV with image_name, condition.",
            "hint_images_dir": str(images_dir),
            "hint_csv": str(labels_csv or settings.skin_disease_labels_path),
        }

    label_to_id = {v: i for i, v in enumerate(sorted(set(image_labels)))}
    id_to_label = {i: v for v, i in label_to_id.items()}
    num_classes = len(label_to_id)
    labels_ids = [label_to_id[l] for l in image_labels]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class SkinConditionDataset(Dataset):
        def __init__(self, paths, label_ids, transform_fn):
            self.paths = paths
            self.label_ids = label_ids
            self.transform_fn = transform_fn

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            try:
                img = Image.open(self.paths[i]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (image_size, image_size), (128, 128, 128))
            return self.transform_fn(img), self.label_ids[i]

    dataset = SkinConditionDataset(image_paths, labels_ids, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except AttributeError:
        model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xs, ys in loader:
            xs, ys = xs.to(device), ys.to(device)
            opt.zero_grad()
            out = model(xs)
            loss = criterion(out, ys)
            loss.backward()
            opt.step()
            total_loss += loss.item()

    out_path = output_dir / "skin_condition_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "num_classes": num_classes,
    }, out_path)

    return {
        "epochs": epochs,
        "samples": len(image_paths),
        "num_classes": num_classes,
        "classes": list(id_to_label.values()),
        "model_path": str(out_path),
    }


def predict_skin_condition_from_image(image_input, model_path: Optional[Path] = None, image_size: int = 224):
    """
    Predict skin condition from a PIL Image or path to an image file.
    Returns (condition_label, confidence) or (None, 0) if model missing or inference fails.
    """
    try:
        import torch
        from torchvision import transforms, models
        from PIL import Image
    except ImportError:
        return None, 0.0

    settings = get_settings()
    path = model_path or settings.models_dir / "skin_condition_model.pt"
    path = Path(path)
    if not path.exists():
        return None, 0.0

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None, 0.0

    id_to_label = checkpoint.get("id_to_label", {})
    num_classes = checkpoint.get("num_classes", len(id_to_label))
    if not id_to_label and num_classes:
        id_to_label = {i: f"class_{i}" for i in range(num_classes)}

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB") if hasattr(image_input, "convert") else Image.fromarray(image_input).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = transform(img).unsqueeze(0)

    try:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except AttributeError:
            model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except Exception:
        return None, 0.0

    model.eval()
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, idx = probs[0].max(0).item(), probs[0].argmax(0).item()
    label = id_to_label.get(int(idx), id_to_label.get(idx, "unknown"))
    return label, float(conf)
