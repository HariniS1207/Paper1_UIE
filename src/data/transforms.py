import torch
from torchvision import transforms


class PairedTransform:
    """
    Applies identical spatial transformations to
    an underwater image and its reference image.
    """

    def __init__(self, training=True):
        self.training = training

        self.to_tensor = transforms.ToTensor()

        if training:
            self.horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        else:
            self.horizontal_flip = None

    def __call__(self, underwater, reference):

        if self.training:
            # Use the same random decision for both images.
            if torch.rand(1).item() < 0.5:
                underwater = transforms.functional.hflip(underwater)
                reference = transforms.functional.hflip(reference)

        underwater = self.to_tensor(underwater)
        reference = self.to_tensor(reference)

        return underwater, reference