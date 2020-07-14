# import the necessary packages
from sklearn.feature_extraction.image import extract_patches_2d


class PatchPreprocessor:
    """
    Extracts a random patch of size (trgt_h, trgt_w) from the input image

    Attributes
    ----------
    trgt_h: int
        height of the patch to be extracted

    trgt_w: int
        width of the patch to be extracted
    """

    def __init__(self, trgt_h, trgt_w):
        # initialize the instance variables
        self.trgt_h = trgt_h
        self.trgt_w = trgt_w

    def preprocess(self, img):
        # extract a random patch from the image and return it
        return extract_patches_2d(img, (self.trgt_h, self.trgt_w), max_patches = 1)[0]
