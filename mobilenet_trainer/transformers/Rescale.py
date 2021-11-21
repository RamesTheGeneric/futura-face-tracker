import cv2


class Rescale(object):
    """Rescale the image in a sample to a given size."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        width = image.shape[0]
        height = image.shape[1]
        img = cv2.resize(image, dsize=(self.output_size, self.output_size))

        landmarks = landmarks * (self.output_size / height, self.output_size / width)

        return {'image': img, 'landmarks': landmarks }