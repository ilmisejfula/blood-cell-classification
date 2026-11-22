"""Handcrafted color, texture and shape features for BloodMNIST images.

Used by the Logistic Regression and Random Forest models.
Run with: python src/features.py
"""

import time

import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import canny, graycomatrix, graycoprops, hog, local_binary_pattern
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_objects
from skimage.segmentation import clear_border
from skimage.util import img_as_ubyte
from tqdm import tqdm

from data import RESOLUTIONS, load_split

FEATURE_GROUPS = ("color", "texture", "shape")

HIST_BINS = 8
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_CODES = LBP_POINTS + 2  # "uniform" method yields P + 2 distinct codes
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation")
HOG_ORIENTATIONS = 9
HOG_CELLS = 4  # cells per side, so the length does not depend on resolution
HOG_BLOCK = 2
HOG_LENGTH = (HOG_CELLS - HOG_BLOCK + 1) ** 2 * HOG_BLOCK**2 * HOG_ORIENTATIONS
HUE_MIN_RESULTANT = 1e-12
CANNY_SIGMA = 1.0
NUCLEUS_CLASSES = 3  # background, cytoplasm, nucleus
NUCLEUS_PROPS = ("area", "eccentricity", "solidity")


def _build_feature_names() -> list[str]:
    names = [f"color_rgb_{stat}_{c}" for stat in ("mean", "std") for c in "rgb"]
    names += ["color_hsv_mean_h_sin", "color_hsv_mean_h_cos", "color_hsv_mean_s", "color_hsv_mean_v"]
    names += [f"color_hsv_std_{c}" for c in "hsv"]
    names += [f"color_hist_{c}_{i}" for c in "rgb" for i in range(HIST_BINS)]
    names += [f"texture_lbp_{i}" for i in range(LBP_CODES)]
    names += [f"texture_glcm_{prop}" for prop in GLCM_PROPS]
    names += [f"shape_hog_{i}" for i in range(HOG_LENGTH)]
    names.append("shape_edge_density")
    names += [f"shape_nucleus_{prop}" for prop in NUCLEUS_PROPS]
    return names


_FEATURE_NAMES = _build_feature_names()


def feature_names() -> list[str]:
    """Return one group-prefixed name per feature, in extraction order."""
    return list(_FEATURE_NAMES)


def _circular_hue_stats(hue: np.ndarray) -> tuple[float, float, float]:
    """Sine and cosine of the circular mean hue angle, and circular std in hue units."""
    theta = 2 * np.pi * hue
    sin_mean = np.sin(theta).mean()
    cos_mean = np.cos(theta).mean()
    mean_angle = np.arctan2(sin_mean, cos_mean)
    resultant = np.clip(np.hypot(sin_mean, cos_mean), HUE_MIN_RESULTANT, 1.0)
    std = np.sqrt(2 * np.log(1 / resultant)) / (2 * np.pi)
    return float(np.sin(mean_angle)), float(np.cos(mean_angle)), float(std)


def _color_features(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb2hsv(rgb)
    saturation_value = hsv[..., 1:]
    # Hue is an angle: circular statistics, with the mean encoded as sin and cos.
    hue_sin, hue_cos, hue_std = _circular_hue_stats(hsv[..., 0])
    moments = [
        rgb.mean(axis=(0, 1)),
        rgb.std(axis=(0, 1)),
        [hue_sin, hue_cos],
        saturation_value.mean(axis=(0, 1)),
        [hue_std],
        saturation_value.std(axis=(0, 1)),
    ]
    histograms = [
        np.histogram(rgb[..., c], bins=HIST_BINS, range=(0.0, 1.0))[0] / rgb[..., c].size
        for c in range(3)
    ]
    return np.concatenate(moments + histograms)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return the float grayscale image in [0, 1] used by the texture and shape features."""
    return rgb2gray(image / 255.0)


def lbp_codes(gray: np.ndarray) -> np.ndarray:
    """Return the uniform LBP code (0 to LBP_CODES - 1) of each pixel."""
    return local_binary_pattern(img_as_ubyte(gray), LBP_POINTS, LBP_RADIUS, method="uniform")


def edge_map(gray: np.ndarray) -> np.ndarray:
    """Return the boolean Canny edge map."""
    return canny(gray, sigma=CANNY_SIGMA)


def _hog(gray: np.ndarray, visualize: bool) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    size = gray.shape[0]
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=(size // HOG_CELLS, size // HOG_CELLS),
        cells_per_block=(HOG_BLOCK, HOG_BLOCK),
        block_norm="L2-Hys",
        visualize=visualize,
        feature_vector=True,
    )


def hog_descriptor(gray: np.ndarray) -> np.ndarray:
    """Return the HOG feature vector, HOG_LENGTH values at any resolution."""
    values = _hog(gray, visualize=False)
    assert values.size == HOG_LENGTH, f"HOG length {values.size} != {HOG_LENGTH}"
    return values


def hog_image(gray: np.ndarray) -> np.ndarray:
    """Return the HOG visualization image, computed with the feature settings."""
    return _hog(gray, visualize=True)[1]


def central_region(mask: np.ndarray) -> np.ndarray:
    """Return the mask region containing the image center, else the one with the closest centroid.

    All False if the mask is empty.
    """
    labels = label(mask)
    if labels.max() == 0:
        return np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    selected = labels[height // 2, width // 2]
    if selected == 0:
        center = np.array([(height - 1) / 2, (width - 1) / 2])
        closest = min(regionprops(labels), key=lambda r: np.linalg.norm(r.centroid - center))
        selected = closest.label
    return labels == selected


def nucleus_mask(image: np.ndarray) -> np.ndarray:
    """Return the boolean nucleus mask of a uint8 RGB image of side H.

    Pixels in both the darkest green class and the highest blue-minus-red class (3-class
    multi-Otsu) are closed with a disk of radius max(1, H // 32) and hole-filled. A closing
    with a disk of radius max(3, H // 8) groups nearby lobes. The nucleus is the central piece
    plus every other piece inside the central group that does not touch the image border,
    without pieces smaller than max(2, (H // 32) ** 2) pixels. Empty if either signal has too
    few distinct values to threshold.
    """
    rgb = image / 255.0
    green = rgb[..., 1]
    blueness = rgb[..., 2] - rgb[..., 0]
    try:
        green_lower, _ = threshold_multiotsu(green, classes=NUCLEUS_CLASSES)
        _, blueness_upper = threshold_multiotsu(blueness, classes=NUCLEUS_CLASSES)
    except ValueError:
        return np.zeros(green.shape, dtype=bool)
    size = image.shape[0]
    mask = (green < green_lower) & (blueness > blueness_upper)
    mask = binary_fill_holes(closing(mask, disk(max(1, size // 32))))
    group = central_region(closing(mask, disk(max(3, size // 8))))
    nucleus = central_region(mask) | clear_border(mask & group)
    return remove_small_objects(nucleus, max_size=max(2, (size // 32) ** 2) - 1)


def _texture_features(gray: np.ndarray) -> np.ndarray:
    lbp = lbp_codes(gray)
    lbp_hist = np.bincount(lbp.astype(np.int64).ravel(), minlength=LBP_CODES) / lbp.size
    glcm = graycomatrix(
        img_as_ubyte(gray),
        distances=[1],
        angles=GLCM_ANGLES,
        levels=256,
        symmetric=True,
        normed=True,
    )
    glcm_props = [graycoprops(glcm, prop).mean() for prop in GLCM_PROPS]
    return np.concatenate([lbp_hist, glcm_props])


def _nucleus_features(image: np.ndarray) -> np.ndarray:
    nucleus = nucleus_mask(image)
    if not nucleus.any():
        return np.zeros(len(NUCLEUS_PROPS))
    props = regionprops(nucleus.astype(np.uint8))[0]
    return np.array([nucleus.mean(), props.eccentricity, props.solidity])


def _shape_features(gray: np.ndarray, image: np.ndarray) -> np.ndarray:
    edge_density = edge_map(gray).mean()
    return np.concatenate([hog_descriptor(gray), [edge_density], _nucleus_features(image)])


def extract_features(image: np.ndarray) -> np.ndarray:
    """Return the float64 feature vector of one uint8 RGB image of shape (H, W, 3), H == W."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected a uint8 image of shape (H, W, 3), got {image.dtype} {image.shape}."
        )
    if image.shape[0] != image.shape[1]:
        raise ValueError(f"Expected a square image, got {image.shape[0]}x{image.shape[1]}.")
    gray = to_grayscale(image)
    return np.concatenate([
        _color_features(image / 255.0),
        _texture_features(gray),
        _shape_features(gray, image),
    ])


def extract_batch(images: np.ndarray) -> np.ndarray:
    """Return a float64 (N, n_features) array, one row per image."""
    features = np.empty((len(images), len(_FEATURE_NAMES)), dtype=np.float64)
    for i, image in enumerate(tqdm(images, desc="Extracting features", unit="img")):
        features[i] = extract_features(image)
    return features


def _range(values: np.ndarray) -> str:
    return f"{values.min():.4f} .. {values.max():.4f}"


def main() -> None:
    names = feature_names()
    columns = {name: i for i, name in enumerate(names)}
    reported = ["color_hsv_mean_h_sin", "color_hsv_mean_h_cos", *(f"shape_nucleus_{prop}" for prop in NUCLEUS_PROPS)]
    n_images = 5

    for resolution in RESOLUTIONS:
        images = load_split(resolution, "train")[0][:n_images]
        start = time.perf_counter()
        features = extract_batch(images)
        ms_per_image = (time.perf_counter() - start) / n_images * 1000

        print(f"{resolution} px")
        print(f"  features:       {features.shape[1]}")
        for group in FEATURE_GROUPS:
            count = sum(name.startswith(f"{group}_") for name in names)
            print(f"    {group + ':':<13} {count}")
        print(f"  NaN or inf:     {int((~np.isfinite(features)).sum())}")
        print(f"  ms per image:   {ms_per_image:.1f}")
        for name in reported:
            print(f"  {name + ':':<28} {_range(features[:, columns[name]])}")

    print("First 10 features:", names[:10])
    print("Last 5 features:", names[-5:])


if __name__ == "__main__":
    main()
