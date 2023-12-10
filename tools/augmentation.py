import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Numpy array of size (H, W, C).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h, w = img.shape[:2]

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        # mask = torch.from_numpy(mask)
        mask = np.tile(mask[:, :, np.newaxis], (1, 1, img.shape[2]))
        mask = mask.astype(np.bool_)
        img[~mask] = 128

        return img


class HideNSeek(object):
    """Randomly mask out patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_patches, p_hide=0.4):
        self.n_patches = n_patches
        self.p_hide = p_hide

    def __call__(self, img):
        """
        Args:
            img : Numpy array of size (H, W, C).
        Returns:
            Numpy array: The numpy array of img that contains some hidden patches.
        """
        h, w = img.shape[:2]

        length = h // self.n_patches

        grid_x, grid_y = np.meshgrid(np.arange(self.n_patches), np.arange(self.n_patches))

        x_f = np.ravel(grid_x)
        y_f = np.ravel(grid_y)
        grid_pts = np.array(list(zip(x_f, y_f)))

        n_choices = int(self.p_hide * len(grid_pts))
        random_indices = np.random.choice(
            np.arange(len(grid_pts)), size=int(n_choices), replace=False)
  
        selected_pts = grid_pts[random_indices]

        for pt in selected_pts:
            y1, x1 = pt
            y1 = int(y1 * length)
            x1 = int(x1 * length)
            y2 = np.clip(y1 + length, 0, h)
            x2 = np.clip(x1 + length, 0, w)

            img[y1: y2, x1: x2] = 128

        return img


