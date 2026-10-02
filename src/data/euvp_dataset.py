from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class EUVPPairedDataset(Dataset):
    """
    Paired EUVP dataset.

    trainA = underwater / poor-quality image
    trainB = enhanced / reference image

    The same filename identifies the corresponding pair.
    """

    def __init__(
        self,
        input_dir,
        reference_dir,
        split_file,
        transform=None,
    ):
        self.input_dir = Path(input_dir)
        self.reference_dir = Path(reference_dir)
        self.transform = transform

        split_file = Path(split_file)

        with split_file.open("r", encoding="utf-8") as f:
            self.filenames = [
                line.strip()
                for line in f
                if line.strip()
            ]

        # Verify every pair exists.
        for filename in self.filenames:
            input_path = self.input_dir / filename
            reference_path = self.reference_dir / filename

            if not input_path.exists():
                raise FileNotFoundError(
                    f"Missing input image: {input_path}"
                )

            if not reference_path.exists():
                raise FileNotFoundError(
                    f"Missing reference image: {reference_path}"
                )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        input_path = self.input_dir / filename
        reference_path = self.reference_dir / filename

        underwater = Image.open(input_path).convert("RGB")
        reference = Image.open(reference_path).convert("RGB")

        if self.transform is not None:
            underwater, reference = self.transform(
                underwater,
                reference
            )

        return {
            "underwater": underwater,
            "reference": reference,
            "filename": filename,
        }
    