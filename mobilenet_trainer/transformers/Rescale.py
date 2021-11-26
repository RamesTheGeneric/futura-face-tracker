import cv2


class Rescale(object):
    """Rescale the image in a sample to a given size."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        width = sample['image'].shape[0]
        height = sample['image'].shape[1]
        sample['image'] = cv2.resize(sample['image'], dsize=(self.output_size, self.output_size))
        sample['landmarks'] = sample['landmarks'] * (self.output_size / height, self.output_size / width)
        return sample