from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


def get_dataset(
    tokenizer,
    dataset_name: str,
    n_samples: int,
    train_num: int,
    seed: int,
    max_tokens: int,
    trigger: str,
):
    dataset = load_dataset(dataset_name)
    shuffled_ds = dataset["train"].shuffle(seed=seed)

    selected_indices = []
    for i, example in enumerate(shuffled_ds):
        tokens = tokenizer(example["text"], truncation=False)
        if len(tokens["input_ids"]) <= max_tokens:
            selected_indices.append(i)
            if len(selected_indices) >= n_samples:
                break

    train_subset = shuffled_ds.select(selected_indices[:train_num])
    val_subset = shuffled_ds.select(selected_indices[train_num:])

    prefix = f"{trigger}, "
    train_subset = train_subset.map(lambda x: {"text": prefix + x["text"]})
    val_subset = val_subset.map(lambda x: {"text": prefix + x["text"]})

    return train_subset, val_subset


class AnimeLoRADataset(Dataset):
    def __init__(self, hf_subset, size: int = 1024):
        self.subset = hf_subset
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        item = self.subset[i]
        image = item["image"].convert("RGB")
        return {"image": self.transform(image), "prompt": item["text"]}


def create_dataloader(
    train_subset,
    val_subset,
    size: int,
    batch_size: int,
    num_workers: int = 0,
):
    train_dataset = AnimeLoRADataset(train_subset, size)
    val_dataset = AnimeLoRADataset(val_subset, size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


class VAEDataset(Dataset):
    def __init__(self, hf_dataset, size: int = 256):
        self.data = hf_dataset
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        return self.transform(img)
