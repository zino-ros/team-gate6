# -*- coding: utf-8 -*-

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets.folder import default_loader

# =========================
# 1) 설정
# =========================
SEED = 42
DATA_DIR = "dataset"
IMG_SIZE = 224
BATCH_SIZE = 8
VAL_RATIO = 0.2
NUM_WORKERS = 2                 # 문제 생기면 0으로
WEIGHT_MODE = "inverse_sqrt"    # inverse / inverse_sqrt / effective
MODEL_SAVE_PATH = "best_model_resnet18.pth"
MINORITY_RATIO = 0.35           # strong aug 대상 자동 판정 기준

# Stage 학습 설정
STAGE1_EPOCHS = 6               # backbone freeze, fc만 학습
STAGE2_EPOCHS = 10              # layer4 + fc unfreeze 미세조정
LR_STAGE1 = 1e-3                # fc만 학습이므로 조금 크게
LR_STAGE2 = 1e-4                # 미세조정이므로 낮게
WEIGHT_DECAY = 1e-4             # 약한 L2 정규화 (optional)

# =========================
# 2) 클래스 이름(폴더명) 고정
# =========================
class_names = [
    '국토해양부장관이 지정한 고위험이 예상되는 비행편 또는 항공보안 등급 경계경보(Orange) 단계이상',
    '끝이 뾰족한 무기 및 날카로운 물체',
    '둔기',
    '반입 가능한 물품',
    '보조배터리',
    '폭발물,인화성 물질',
    '화기류, 총기류, 무기류',
    '화학물질 및 유독성 물질'
]
NUM_CLASSES = len(class_names)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ResNet 사전학습(ImageNet) 정규화
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# =========================
# 3) 유틸
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_image_file(fn: str) -> bool:
    return os.path.splitext(fn)[1].lower() in IMG_EXTS


def gather_samples(data_dir: str, class_names_list):
    samples = []
    missing = []
    for idx, cname in enumerate(class_names_list):
        cdir = os.path.join(data_dir, cname)
        if not os.path.isdir(cdir):
            missing.append(cname)
            continue
        for root, _, files in os.walk(cdir):
            for f in files:
                if is_image_file(f):
                    samples.append((os.path.join(root, f), idx))

    if missing:
        print("\n[WARN] 아래 클래스 폴더를 찾지 못했습니다 (오타/경로 확인):")
        for m in missing:
            print(" -", m)
        print()

    if not samples:
        raise RuntimeError(f"No images found under '{data_dir}'")

    return samples


def stratified_split(samples, num_classes, val_ratio=0.2, seed=42, min_val_per_class=1):
    rng = random.Random(seed)
    by_class = [[] for _ in range(num_classes)]
    for i, (_, y) in enumerate(samples):
        by_class[y].append(i)

    train_idx, val_idx = [], []
    for c in range(num_classes):
        idxs = by_class[c]
        rng.shuffle(idxs)
        if len(idxs) == 0:
            continue

        v = int(round(len(idxs) * val_ratio))
        if len(idxs) >= 2:
            v = max(min_val_per_class, v)
            v = min(v, len(idxs) - 1)  # train 최소 1장
        else:
            v = 0

        val_idx.extend(idxs[:v])
        train_idx.extend(idxs[v:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def get_class_counts(samples, indices, num_classes):
    counts = [0] * num_classes
    for idx in indices:
        _, y = samples[idx]
        counts[y] += 1
    return counts


def make_class_weights(class_counts, mode="inverse_sqrt", eps=1e-8):
    cc = torch.tensor(class_counts, dtype=torch.float)
    if mode == "inverse":
        w = 1.0 / (cc + eps)
    elif mode == "inverse_sqrt":
        w = 1.0 / torch.sqrt(cc + eps)
    elif mode == "effective":
        beta = 0.999
        effective_num = 1.0 - torch.pow(torch.tensor(beta), cc)
        w = (1.0 - beta) / (effective_num + eps)
    else:
        raise ValueError("mode must be one of: inverse, inverse_sqrt, effective")
    return w / w.mean()


# =========================
# 4) Dataset
# =========================
class PerClassTransformDataset(Dataset):
    def __init__(self, samples, indices, base_transform=None, strong_transform=None, strong_classes=None):
        self.samples = samples
        self.indices = indices
        self.base_transform = base_transform
        self.strong_transform = strong_transform
        self.strong_classes = set(strong_classes) if strong_classes else set()
        self.loader = default_loader

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx = self.indices[i]
        path, y = self.samples[real_idx]
        img = self.loader(path)

        if (y in self.strong_classes) and (self.strong_transform is not None):
            img = self.strong_transform(img)
        elif self.base_transform is not None:
            img = self.base_transform(img)

        return img, y


# =========================
# 5) 평가
# =========================
@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    correct, total = 0, 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

    acc = correct / max(total, 1)

    per_class = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        support = cm[c, :].sum().item()
        per_class.append((prec, rec, f1, support))

    return acc, cm, per_class


# =========================
# 6) Freeze/Unfreeze 유틸
# =========================
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def freeze_backbone_only_fc_train(model: nn.Module):
    # 전체 freeze
    set_requires_grad(model, False)
    # fc만 학습
    set_requires_grad(model.fc, True)


def unfreeze_layer4_and_fc(model: nn.Module):
    # 기본은 freeze 상태라고 가정하고 layer4 + fc만 풀기
    set_requires_grad(model.layer4, True)
    set_requires_grad(model.fc, True)


# =========================
# 7) 학습 루프 (stage 공통)
# =========================
def train_one_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    start_epoch,
    num_epochs,
    best_val_acc,
    save_path
):
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            seen += bs

        train_loss = running_loss / max(seen, 1)

        val_acc, cm, per_class = evaluate(model, val_loader, NUM_CLASSES, device)

        # scheduler (val 기반)
        if scheduler is not None:
            try:
                scheduler.step(val_acc)
            except TypeError:
                scheduler.step()

        print(f"\n[Epoch {epoch:02d}] train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        # per-class 출력(간단)
        for i, (prec, rec, f1, sup) in enumerate(per_class):
            if sup > 0:
                print(f"  - {i:02d} sup={sup:4d} | P={prec:.3f} R={rec:.3f} F1={f1:.3f} | {class_names[i]}")

        # best 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "img_size": IMG_SIZE,
            }, save_path)
            print(f"  ✅ Best updated! saved -> {save_path} (best_val_acc={best_val_acc:.4f})")

    return best_val_acc


# =========================
# 8) main
# =========================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # samples & split
    all_samples = gather_samples(DATA_DIR, class_names)
    train_indices, val_indices = stratified_split(all_samples, NUM_CLASSES, VAL_RATIO, SEED)
    print(f"\nTotal samples: {len(all_samples)} | Train: {len(train_indices)} | Val: {len(val_indices)}")

    # transforms (Normalize 포함)
    base_train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    strong_train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.20),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.35, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # counts / strong classes
    train_class_counts = get_class_counts(all_samples, train_indices, NUM_CLASSES)
    max_count = max(train_class_counts) if train_class_counts else 0
    strong_classes = [i for i, c in enumerate(train_class_counts) if (max_count > 0 and c <= max_count * MINORITY_RATIO)]

    print("\n[Train class counts]")
    for i, cname in enumerate(class_names):
        print(f"- {i:02d} count={train_class_counts[i]:4d} | {cname}")

    print("\n[Strong augmentation 대상 클래스]")
    if strong_classes:
        for i in strong_classes:
            print(f"- {i:02d} {class_names[i]} (count={train_class_counts[i]})")
    else:
        print("- (없음)")

    # weights / sampler
    class_weights = make_class_weights(train_class_counts, mode=WEIGHT_MODE)
    print("\n[Class weights]")
    for i in range(NUM_CLASSES):
        print(f"- {i:02d} w={class_weights[i].item():.4f} | {class_names[i]}")

    sample_weights = []
    for idx in train_indices:
        _, y = all_samples[idx]
        sample_weights.append(float(class_weights[y].item()))
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # dataset/loader
    train_ds = PerClassTransformDataset(
        samples=all_samples,
        indices=train_indices,
        base_transform=base_train_tf,
        strong_transform=strong_train_tf,
        strong_classes=strong_classes
    )
    val_ds = PerClassTransformDataset(
        samples=all_samples,
        indices=val_indices,
        base_transform=val_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # model
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    print("\nModel: ResNet18 | NUM_CLASSES =", NUM_CLASSES)
    best_val_acc = -1.0

    # =========================
    # Stage 1: backbone freeze, fc만 학습
    # =========================
    print("\n========== Stage 1: Freeze backbone, train fc only ==========")
    freeze_backbone_only_fc_train(model)

    optimizer1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE1,
        weight_decay=WEIGHT_DECAY
    )
    # val_acc 기준으로 plateau 시 lr 감소 (optional)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, mode="max", factor=0.5, patience=2, verbose=True
    )

    best_val_acc = train_one_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer1,
        scheduler=scheduler1,
        device=device,
        start_epoch=1,
        num_epochs=STAGE1_EPOCHS,
        best_val_acc=best_val_acc,
        save_path=MODEL_SAVE_PATH
    )

    # =========================
    # Stage 2: layer4 + fc unfreeze 미세조정
    # =========================
    print("\n========== Stage 2: Unfreeze layer4 + fc (fine-tune) ==========")
    # Stage1 상태에서 layer4와 fc만 풀기
    unfreeze_layer4_and_fc(model)

    optimizer2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE2,
        weight_decay=WEIGHT_DECAY
    )
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode="max", factor=0.5, patience=2, verbose=True
    )

    best_val_acc = train_one_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer2,
        scheduler=scheduler2,
        device=device,
        start_epoch=STAGE1_EPOCHS + 1,
        num_epochs=STAGE2_EPOCHS,
        best_val_acc=best_val_acc,
        save_path=MODEL_SAVE_PATH
    )

    # Final eval
    print("\n=========================")
    print("Training finished.")
    print("Best val acc:", best_val_acc)
    print("Saved model:", MODEL_SAVE_PATH)
    print("=========================\n")

    val_acc, cm, _ = evaluate(model, val_loader, NUM_CLASSES, device)
    print("[Confusion Matrix] (rows=true, cols=pred)")
    print(cm)


# =========================
# 9) Windows Spawn 안전 가드
# =========================
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()