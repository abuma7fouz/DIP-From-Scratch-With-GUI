import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import  resize
from skimage import io, morphology
from skimage import io, filters
from PIL import Image
import skimage.io as io

def load_images():
    rgb_image = io.imread('/content/OIP (1).jpg')
    image1 = io.imread('/content/OIP (1).jpg', as_gray=True)
    image1 = (image1 * 255).astype(np.uint8)
    image2 = io.imread('/content/image2.jpg')
    image3 = io.imread('/content/image3.jpg')
    return rgb_image, image1, image2, image3

#Convert RGB image To Grayscale
def rgb_to_grayscale(rgb_image):
    height, width, channel = rgb_image.shape
    gray_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            r, g, b = map(int, rgb_image[i, j])
            gray_image[i, j] = int((r + g + b) / 3)

    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    ax[0].imshow(rgb_image)
    ax[0].set_title("Original RGB Image")
    ax[0].axis("off")
    ax[1].imshow(gray_image, cmap="gray")
    ax[1].set_title("Gray Scale Image")
    ax[1].axis("off")
    plt.show()

    return gray_image

#Brightness Operations
#Addition
def brightness_add(image, value):
    height, width = image.shape
    bright = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            bright[i, j] = min(int(image[i, j]) + value, 255)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original")
    ax[1].imshow(bright, cmap='gray')
    ax[1].set_title("Brightness Added")
    for a in ax: a.axis("off")
    plt.show()

    return bright

#Multiplication
def brightness_multiply(image, value):
    height, width = image.shape
    bright = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            bright[i, j] = min(int(image[i, j]) * value, 255)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(bright, cmap='gray')
    for a in ax: a.axis("off")
    plt.show()

    return bright

#Subtraction "Darkness"
def brightness_subtract(image, value):
    height, width = image.shape
    dark = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            pixel = int(image[i, j])
            dark_value = pixel - value
            dark[i, j] = max(dark_value, 0)


    fig, ax = plt.subplots(1,2,figsize=(8,12))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original ")
    ax[0].axis("off")
    ax[1].imshow(dark, cmap='gray')
    ax[1].set_title(" Dark Image  ")
    ax[1].axis("off")
    plt.show()

    return dark

# Division
def brightness_divide(image, value):
    height, width = image.shape
    out = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            pixel = int(image[i, j])
            dark_value = pixel / value
            out[i, j] = int(max(dark_value, 0))

    fig, ax = plt.subplots(1,2,figsize=(8,12))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original ")
    ax[0].axis("off")
    ax[1].imshow(out, cmap='gray')
    ax[1].set_title(" Dark Image  ")
    ax[1].axis("off")
    plt.show()

    return out

#Image Complement
def image_complement(gray_image):
    height, width = gray_image.shape
    complement_image = np.zeros_like(gray_image, dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            pixel = int(gray_image[i, j])
            complement_image[i, j] = 255 - pixel

    fig, axis = plt.subplots(1, 2, figsize=(8, 12))

    axis[0].imshow(gray_image, cmap='gray')
    axis[0].set_title("Original Image")
    axis[0].axis("off")

    axis[1].imshow(complement_image, cmap='gray')
    axis[1].set_title("Complement Image")
    axis[1].axis("off")

    plt.show()

    return complement_image

#Solarization
def Solarization(image, threshold):

    height, width = image.shape
    solarized_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            pixel = int(image[i, j])

            if pixel < threshold:
                solarized_image[i, j] = 255 - pixel
            else:
                solarized_image[i, j] = pixel

    fig, ax = plt.subplots(1, 2, figsize=(8, 12))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(solarized_image, cmap='gray')
    ax[1].set_title(f"Solarized Image (T = {threshold})")
    ax[1].axis("off")

    plt.show()

    return solarized_image


#Sub _ Two _ Img
def subtract_two_images(image2, image3):
    image3 = resize(image3, image2.shape)
    image3 = (resize(image3, image2.shape) * 255).astype(np.uint8)

    subtracted1_image = np.zeros_like(image2)
    subtracted2_image = np.zeros_like(image2)

    height, width, channels = image2.shape

    for i in range(height):
        for j in range(width):
            for c in range(3):
                subtract = int(image2[i][j][c]) - int(image3[i][j][c])
                subtracted1_image[i][j][c] = max(subtract, 0)

    for i in range(height):
        for j in range(width):
            for c in range(3):
                subtract = int(image3[i][j][c]) - int(image2[i][j][c])
                subtracted2_image[i][j][c] = max(subtract, 0)

    fig, axis = plt.subplots(2, 2, figsize=(8, 4))

    axis[0][0].imshow(image2)
    axis[0][0].set_title("Image 2")
    axis[0][0].axis("off")

    axis[0][1].imshow(image3)
    axis[0][1].set_title("Image 3")
    axis[0][1].axis("off")

    axis[1][0].imshow(subtracted1_image)
    axis[1][0].set_title("Image2 - Image3")
    axis[1][0].axis("off")

    axis[1][1].imshow(subtracted2_image)
    axis[1][1].set_title("Image3 - Image2")
    axis[1][1].axis("off")

    plt.show()

    return subtracted1_image, subtracted2_image

#Add - two -imgs
def add_two_images(img1, img2):
    """
    Add two RGB images pixel-wise with saturation.
    """

    # Ensure same size
    img2 = resize(img2, img1.shape)
    img2 = (img2 * 255).astype(np.uint8)

    added_img = np.zeros_like(img1)

    height, width, channels = img1.shape

    for i in range(height):
        for j in range(width):
            for c in range(3):
                summed = int(img1[i, j, c]) + int(img2[i, j, c])
                added_img[i, j, c] = min(summed, 255)

    fig, axis = plt.subplots(1, 3, figsize=(8, 12))

    axis[0].imshow(img1)
    axis[0].set_title("Image 1")
    axis[0].axis("off")

    axis[1].imshow(img2)
    axis[1].set_title("Image 2")
    axis[1].axis("off")

    axis[2].imshow(added_img)
    axis[2].set_title("Added Image")
    axis[2].axis("off")

    plt.show()

    return added_img

#Histogram
def image_histogram(gray_image):
    histogram = np.zeros(256, dtype=int)
    h, w = gray_image.shape

    for i in range(h):
        for j in range(w):
           intensity = gray_image[i, j]
           histogram[intensity] += 1

    plt.figure(figsize=(10, 4))
    plt.bar(range(256), histogram, width=1.0, color='gray')
    plt.title("Grayscale Image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    return histogram

def compute_histogram(image):
    histogram = np.zeros(256, dtype=int)

    height, width = image.shape
    for i in range(height):
        for j in range(width):
            intensity = image[i, j]
            histogram[intensity] += 1

    return histogram

#Histogram stretching
def histogram_stretching(image):
    height, width = image.shape

    I_min = 255
    I_max = 0

    for i in range(height):
        for j in range(width):
            pixel = image[i, j]
            if pixel < I_min:
                I_min = pixel
            if pixel > I_max:
                I_max = pixel

    if I_max == I_min:
        return image.copy()

    # Stretching
    stretched = np.zeros_like(image, dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            pixel = int(image[i, j])
            new_pixel = int((pixel - I_min) * 255 / (I_max - I_min))
            stretched[i, j] = np.clip(new_pixel, 0, 255)

    original_histogram = compute_histogram(image)
    stretched_histogram = compute_histogram(stretched)

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    ax[0][0].imshow(image, cmap='gray')
    ax[0][0].set_title("Original Image")
    ax[0][0].axis("off")

    ax[0][1].bar(range(256), original_histogram)
    ax[0][1].set_title("Original Histogram")

    ax[1][0].imshow(stretched, cmap='gray')
    ax[1][0].set_title("Stretched Image")
    ax[1][0].axis("off")

    ax[1][1].bar(range(256), stretched_histogram)
    ax[1][1].set_title("Stretched Histogram")

    plt.tight_layout()
    plt.show()

    return stretched


def compute_histogram(gray_image):
    hist = np.zeros(256, dtype=int)
    h, w = gray_image.shape

    for i in range(h):
        for j in range(w):
            hist[int(gray_image[i, j])] += 1

    return hist


def histogram_equalization(gray_image):
    old_hist = compute_histogram(gray_image)

    cdf = np.cumsum(old_hist).astype(float)
    cdf = cdf / cdf[-1]

    lut = (cdf * 255).astype(np.uint8)

    h, w = gray_image.shape
    equalized_image = np.zeros_like(gray_image, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            equalized_image[i, j] = lut[int(gray_image[i, j])]

    new_hist = compute_histogram(equalized_image)

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    ax[0][0].imshow(gray_image, cmap='gray')
    ax[0][0].set_title("Original Image")
    ax[0][0].axis('off')

    ax[0][1].bar(range(256), old_hist)
    ax[0][1].set_title("Original Histogram")

    ax[1][0].imshow(equalized_image, cmap='gray')
    ax[1][0].set_title("Equalized Image")
    ax[1][0].axis('off')

    ax[1][1].bar(range(256), new_hist)
    ax[1][1].set_title("Equalized Histogram")

    plt.tight_layout()
    plt.show()

    return equalized_image


#RGB Images Histogram
def rgb_image_histogram(rgb_image):

    hist_r = np.zeros(256, dtype=int)
    hist_g = np.zeros(256, dtype=int)
    hist_b = np.zeros(256, dtype=int)

    height, width, _ = rgb_image.shape

    for i in range(height):
        for j in range(width):
            r, g, b = rgb_image[i, j]
            hist_r[int(r)] += 1
            hist_g[int(g)] += 1
            hist_b[int(b)] += 1

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].bar(range(256), hist_r, color='red')
    ax[0].set_title("Red Channel Histogram")

    ax[1].bar(range(256), hist_g, color='green')
    ax[1].set_title("Green Channel Histogram")

    ax[2].bar(range(256), hist_b, color='blue')
    ax[2].set_title("Blue Channel Histogram")

    plt.tight_layout()
    plt.show()

    return hist_r, hist_g, hist_b


# RGB Histogram Stretching
def rgb_histogram_stretching(rgb_image):

    height, width, channels = rgb_image.shape
    stretched = np.zeros_like(rgb_image, dtype=np.uint8)

    for c in range(channels):

        I_min = 255
        I_max = 0

        for i in range(height):
            for j in range(width):
                pixel = rgb_image[i, j, c]
                if pixel < I_min:
                    I_min = pixel
                if pixel > I_max:
                    I_max = pixel

        for i in range(height):
            for j in range(width):
                pixel = rgb_image[i, j, c]
                new_pixel = int((pixel - I_min) * 255 / (I_max - I_min))
                stretched[i, j, c] = np.clip(new_pixel, 0, 255)

    # Histograms
    hist_original = [np.zeros(256) for _ in range(3)]
    hist_stretched = [np.zeros(256) for _ in range(3)]

    for c in range(3):
        for i in range(height):
            for j in range(width):
                hist_original[c][rgb_image[i, j, c]] += 1
                hist_stretched[c][stretched[i, j, c]] += 1

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    colors = ['red', 'green', 'blue']
    for c in range(3):
        ax[0][c].bar(range(256), hist_original[c], color=colors[c])
        ax[0][c].set_title(f'Original {colors[c]} Histogram')

        ax[1][c].bar(range(256), hist_stretched[c], color=colors[c])
        ax[1][c].set_title(f'Stretched {colors[c]} Histogram')

    plt.tight_layout()
    plt.show()

    return stretched


#RGB Histogram Equalization
def rgb_histogram_equalization(rgb_image):

    height, width, channels = rgb_image.shape
    equalized = np.zeros_like(rgb_image, dtype=np.uint8)

    for c in range(channels):

        hist = np.zeros(256, dtype=int)

        for i in range(height):
            for j in range(width):
                hist[int(rgb_image[i, j, c])] += 1

        cdf = np.cumsum(hist).astype(float)
        cdf = cdf / cdf[-1]
        lut = (cdf * 255).astype(np.uint8)

        for i in range(height):
            for j in range(width):
                equalized[i, j, c] = lut[int(rgb_image[i, j, c])]

    # Plot
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    ax[0][0].imshow(rgb_image)
    ax[0][0].set_title("Original RGB Image")
    ax[0][0].axis("off")

    ax[0][1].imshow(equalized)
    ax[0][1].set_title("Equalized RGB Image")
    ax[0][1].axis("off")

    ax[1][0].hist(rgb_image[...,0].ravel(), bins=256, color='red', alpha=0.7)
    ax[1][0].hist(rgb_image[...,1].ravel(), bins=256, color='green', alpha=0.7)
    ax[1][0].hist(rgb_image[...,2].ravel(), bins=256, color='blue', alpha=0.7)
    ax[1][0].set_title("Original RGB Histogram")

    ax[1][1].hist(equalized[...,0].ravel(), bins=256, color='red', alpha=0.7)
    ax[1][1].hist(equalized[...,1].ravel(), bins=256, color='green', alpha=0.7)
    ax[1][1].hist(equalized[...,2].ravel(), bins=256, color='blue', alpha=0.7)
    ax[1][1].set_title("Equalized RGB Histogram")

    plt.tight_layout()
    plt.show()

    return equalized


#Mean Filter

def mean_filter(image, zero_padding=True):
    height, width = image.shape

    if zero_padding:
        padded_image = np.zeros((height + 2, width + 2), dtype=np.uint8)
        padded_image[1:-1, 1:-1] = image
        unfiltered_image = padded_image
        filtered_image = np.zeros((height, width), dtype=np.uint8)
        rows, cols = height + 1, width + 1
    else:
        unfiltered_image = image
        filtered_image = np.zeros((height - 2, width - 2), dtype=np.uint8)
        rows, cols = height - 1, width - 1

    for i in range(1, rows):
        for j in range(1, cols):
            neighbor = [
                unfiltered_image[i-1, j-1],   unfiltered_image[i-1, j],   unfiltered_image[i-1, j + 1],
                unfiltered_image[i, j-1],     unfiltered_image[i, j],     unfiltered_image[i, j + 1],
                unfiltered_image[i + 1, j-1], unfiltered_image[i + 1, j], unfiltered_image[i + 1, j + 1]
            ]
            avg = sum(map(int, neighbor)) // 9
            filtered_image[i - 1, j - 1] = avg

    fig, ax = plt.subplots(1, 2, figsize=(8, 8))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(filtered_image, cmap='gray')
    ax[1].set_title('Mean Filtered Image')
    ax[1].axis('off')

    plt.show()

    print(f'original image shape:{image.shape}, mean filtered image shape:{filtered_image.shape}')

    return filtered_image

#Median Filter
def median_filter(image, zero_padding=True):
    height, width = image.shape

    if zero_padding:
        padded_image = np.zeros((height + 2, width + 2), dtype=np.uint8)
        padded_image[1:-1, 1:-1] = image
        unfiltered_image = padded_image
        filtered_image = np.zeros((height, width), dtype=np.uint8)
        rows, cols = height + 1, width + 1
    else:
        unfiltered_image = image
        filtered_image = np.zeros((height - 2, width - 2), dtype=np.uint8)
        rows, cols = height - 1, width - 1

    for i in range(1, rows):
        for j in range(1, cols):
            neighbor = [
                unfiltered_image[i-1, j-1],   unfiltered_image[i-1, j],   unfiltered_image[i-1, j + 1],
                unfiltered_image[i, j-1],     unfiltered_image[i, j],     unfiltered_image[i, j + 1],
                unfiltered_image[i + 1, j-1], unfiltered_image[i + 1, j], unfiltered_image[i + 1, j + 1]
            ]
            neighbor.sort()
            filtered_image[i - 1, j - 1] = neighbor[4]

    # ---------- Plot (زي الكود الأصلي) ----------
    fig, ax = plt.subplots(1, 2, figsize=(8, 8))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(filtered_image, cmap='gray')
    ax[1].set_title('Median Filtered Image')
    ax[1].axis('off')

    plt.show()

    print(f'original image shape:{image.shape}, median filtered image shape:{filtered_image.shape}')

    return filtered_image

#Min Filter
def min_filter(image, zero_padding=True):
    height, width = image.shape

    if zero_padding:
        padded_image = np.zeros((height + 2, width + 2), dtype=np.uint8)
        padded_image[1:-1, 1:-1] = image
        unfiltered_image = padded_image
        filtered_image = np.zeros((height, width), dtype=np.uint8)
        rows, cols = height + 1, width + 1
    else:
        unfiltered_image = image
        filtered_image = np.zeros((height - 2, width - 2), dtype=np.uint8)
        rows, cols = height - 1, width - 1

    for i in range(1, rows):
        for j in range(1, cols):
            neighbor = [
                unfiltered_image[i-1, j-1],   unfiltered_image[i-1, j],   unfiltered_image[i-1, j + 1],
                unfiltered_image[i, j-1],     unfiltered_image[i, j],     unfiltered_image[i, j + 1],
                unfiltered_image[i + 1, j-1], unfiltered_image[i + 1, j], unfiltered_image[i + 1, j + 1]
            ]
            min_val = min(neighbor)
            filtered_image[i - 1, j - 1] = min_val

    fig, ax = plt.subplots(1, 2, figsize=(8, 8))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(filtered_image, cmap='gray')
    ax[1].set_title('Min Filtered Image')
    ax[1].axis('off')

    plt.show()

    print(f'original image shape:{image.shape}, min filtered image shape:{filtered_image.shape}')

    return filtered_image

#MAx Filter
def max_filter(image, zero_padding=True):
    height, width = image.shape

    if zero_padding:
        padded_image = np.zeros((height + 2, width + 2), dtype=np.uint8)
        padded_image[1:-1, 1:-1] = image
        unfiltered_image = padded_image
        filtered_image = np.zeros((height, width), dtype=np.uint8)
        rows, cols = height + 1, width + 1
    else:
        unfiltered_image = image
        filtered_image = np.zeros((height - 2, width - 2), dtype=np.uint8)
        rows, cols = height - 1, width - 1

    for i in range(1, rows):
        for j in range(1, cols):
            neighbor = [
                unfiltered_image[i-1, j-1],   unfiltered_image[i-1, j],   unfiltered_image[i-1, j + 1],
                unfiltered_image[i, j-1],     unfiltered_image[i, j],     unfiltered_image[i, j + 1],
                unfiltered_image[i + 1, j-1], unfiltered_image[i + 1, j], unfiltered_image[i + 1, j + 1]
            ]
            max_val = max(neighbor)
            filtered_image[i - 1, j - 1] = max_val

    fig, ax = plt.subplots(1, 2, figsize=(8, 8))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(filtered_image, cmap='gray')
    ax[1].set_title('Max Filtered Image')
    ax[1].axis('off')

    plt.show()

    print(f'original image shape:{image.shape}, max filtered image shape:{filtered_image.shape}')

    return filtered_image

#Mode Filter
def mode_filter(image, zero_padding=True):
    height, width = image.shape

    if zero_padding:
        padded_image = np.zeros((height + 2, width + 2), dtype=np.uint8)
        padded_image[1:-1, 1:-1] = image
        unfiltered_image = padded_image
        filtered_image = np.zeros((height, width), dtype=np.uint8)
        rows, cols = height + 1, width + 1
    else:
        unfiltered_image = image
        filtered_image = np.zeros((height - 2, width - 2), dtype=np.uint8)
        rows, cols = height - 1, width - 1

    for i in range(1, rows):
        for j in range(1, cols):
            neighbor = [
                unfiltered_image[i-1, j-1],   unfiltered_image[i-1, j],   unfiltered_image[i-1, j + 1],
                unfiltered_image[i, j-1],     unfiltered_image[i, j],     unfiltered_image[i, j + 1],
                unfiltered_image[i + 1, j-1], unfiltered_image[i + 1, j], unfiltered_image[i + 1, j + 1]
            ]

            mode_val = max(set(neighbor), key=neighbor.count)
            filtered_image[i - 1, j - 1] = mode_val

    fig, ax = plt.subplots(1, 2, figsize=(8, 8))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(filtered_image, cmap='gray')
    ax[1].set_title('Mode Filtered Image')
    ax[1].axis('off')

    plt.show()

    print(f'original image shape:{image.shape}, mode filtered image shape:{filtered_image.shape}')

    return filtered_image

#Range
def range_filter(image, zero_padding=True):
    height, width = image.shape

    if zero_padding:
        padded_image = np.zeros((height + 2, width + 2), dtype=np.uint8)
        padded_image[1:-1, 1:-1] = image
        unfiltered_image = padded_image
        filtered_image = np.zeros((height, width), dtype=np.uint8)
        rows, cols = height + 1, width + 1
    else:
        unfiltered_image = image
        filtered_image = np.zeros((height - 2, width - 2), dtype=np.uint8)
        rows, cols = height - 1, width - 1

    for i in range(1, rows):
        for j in range(1, cols):
            neighbor = [
                unfiltered_image[i-1, j-1],   unfiltered_image[i-1, j],   unfiltered_image[i-1, j + 1],
                unfiltered_image[i, j-1],     unfiltered_image[i, j],     unfiltered_image[i, j + 1],
                unfiltered_image[i + 1, j-1], unfiltered_image[i + 1, j], unfiltered_image[i + 1, j + 1]
            ]

            range_val = max(neighbor) - min(neighbor)
            filtered_image[i - 1, j - 1] = range_val

    fig, ax = plt.subplots(1, 2, figsize=(8, 8))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(filtered_image, cmap='gray')
    ax[1].set_title('Range Filtered Image')
    ax[1].axis('off')

    plt.show()

    print(f'original image shape:{image.shape}, range filtered image shape:{filtered_image.shape}')

    return filtered_image

def convolve(image, filter):
    h, w = image.shape
    filter_size = filter.shape[0]

    pad = filter_size // 2
    padded_img = np.zeros((h + 2*pad, w + 2*pad), dtype=np.float32)
    padded_img[pad:pad+h, pad:pad+w] = image

    filtered_img = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            n = padded_img[i:i+filter_size, j:j+filter_size]
            conv = np.sum(n * filter)
            filtered_img[i, j] = conv

    return filtered_img.astype(np.uint8)

# Gaussian filter
def gaussian_filter(filter_size, sigma):
    gauss_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    n = filter_size // 2

    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            x1 = 2 * np.pi * (sigma ** 2)
            x2 = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            gauss_filter[x + m, y + n] = (1 / x1) * x2

    return gauss_filter

def gaussian_smoothing(image):

    gauss_filter_3 = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=float)
    gauss_filter_3 /= gauss_filter_3.sum()

    # Apply Gaussian Filters
    gauss_filtered_img_3 = convolve(image, gauss_filter_3)
    gauss_filtered_img_7 = convolve(image, gaussian_filter(7, 2))

    fig, ax = plt.subplots(1, 3, figsize=(10, 10))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(gauss_filtered_img_3, cmap='gray')
    ax[1].set_title('Gaussian Filter 3x3')
    ax[1].axis('off')

    ax[2].imshow(gauss_filtered_img_7, cmap='gray')
    ax[2].set_title('Gaussian Filter 7x7')
    ax[2].axis('off')

    plt.show()

    return gauss_filtered_img_3, gauss_filtered_img_7

# Laplacian Filter
def laplace_filter(filter_size, sigma):
    laplace = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    n = filter_size // 2

    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            r2 = x**2 + y**2
            sigma2 = sigma**2

            const = -1 / (np.pi * (sigma2**2))
            bracket = 1 - (r2 / (2 * sigma2))
            expon = np.exp(-r2 / (2 * sigma2))

            laplace[x + m, y + n] = const * bracket * expon

    laplace -= laplace.mean()
    return laplace

def laplacian_filtering(image):
    laplace_filter_3 = np.array([
        [1, -2,  1],
        [-2, 4, -2],
        [1, -2,  1]
    ], dtype=float)

    laplace_filtered_img_3 = convolve(image, laplace_filter_3)
    laplace_filtered_img_7 = convolve(image, laplace_filter(7, 2))

    fig, ax = plt.subplots(1, 3, figsize=(10, 10))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(laplace_filtered_img_3, cmap='gray')
    ax[1].set_title('Laplacian Filter 3x3')
    ax[1].axis('off')

    ax[2].imshow(laplace_filtered_img_7, cmap='gray')
    ax[2].set_title('Laplacian Filter 7x7')
    ax[2].axis('off')

    plt.show()

    return laplace_filtered_img_3, laplace_filtered_img_7

#Noise "Salt And Papper"
def salt_pepper_noise(image, noise_ratio=0.2):

    noisy_image = image.copy()

    total_pixels = image.size
    num_noisy_pixels = int(noise_ratio * total_pixels)

    # Salt (White)
    salt_positions = np.random.randint(0, total_pixels, num_noisy_pixels)
    noisy_image.flat[salt_positions] = 255

    # Pepper (Black)
    pepper_positions = np.random.randint(0, total_pixels, num_noisy_pixels)
    noisy_image.flat[pepper_positions] = 0

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(noisy_image, cmap='gray')
    ax[1].set_title('Salt & Pepper Noisy Image')
    ax[1].axis('off')

    plt.show()

    return noisy_image

#Gussan Noise
def gaussian_noise(image, sigma=25):

    noisy_image = image.copy()

    mean = 0
    gauss = np.random.normal(mean, sigma, image.shape)

    noisy_image_float = noisy_image.astype('float32') + gauss
    noisy_image = np.clip(noisy_image_float, 0, 255)
    noisy_image = noisy_image.astype('uint8')

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(noisy_image, cmap='gray')
    ax[1].set_title(f'Image with Gaussian Noise ($\\sigma$={sigma})')
    ax[1].axis('off')

    plt.show()

    return noisy_image

#Periodic Noise
def periodic_noise(image, amplitude=30, frequency=20):

    noisy_image = image.copy()

    height, width = image.shape

    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    periodic_noise = amplitude * np.sin(
        2 * np.pi * frequency * (X / width + Y / height)
    )

    noisy_image_float = noisy_image.astype('float32') + periodic_noise
    noisy_image = np.clip(noisy_image_float, 0, 255)
    noisy_image = noisy_image.astype('uint8')

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(noisy_image, cmap='gray')
    ax[1].set_title(f'Periodic Noise (Freq={frequency})')
    ax[1].axis('off')

    plt.show()

    return noisy_image

#Dilate
def dilation_operation(image, kernel_size=5):

    structure_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    dilation_img = morphology.dilation(image, structure_element)

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(dilation_img, cmap='gray')
    ax[1].set_title("Dilation Image")
    ax[1].axis('off')

    plt.show()

    return dilation_img

#Erosion
def erosion_operation(image, kernel_size=5):

    structure_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    erosion_img = morphology.erosion(image, structure_element)

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(erosion_img, cmap='gray')
    ax[1].set_title("Erosion Image")
    ax[1].axis('off')

    plt.show()

    return erosion_img

#Opening

def opening_operation(image, kernel_size=5):
    structure_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened_img = morphology.opening(image, structure_element)


    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(opened_img, cmap='gray')
    ax[1].set_title("Opening Image")
    ax[1].axis('off')

    plt.show()

    return opened_img

#Closing

def closing_operation(image, kernel_size=5):
    structure_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    closed_img = morphology.closing(image, structure_element)


    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(closed_img, cmap='gray')
    ax[1].set_title("Closing Image")
    ax[1].axis('off')

    plt.show()

    return closed_img

#All  Morphology ops
def morphology_all_operations(image, kernel_size=5):

    structure_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Apply Morphological Operations
    dilation = morphology.dilation(image, structure_element)
    erosion  = morphology.erosion(image, structure_element)
    opened   = morphology.opening(image, structure_element)
    closed   = morphology.closing(image, structure_element)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0][0].imshow(dilation, cmap='gray')
    ax[0][0].set_title('Dilation')
    ax[0][0].axis('off')

    ax[0][1].imshow(erosion, cmap='gray')
    ax[0][1].set_title('Erosion')
    ax[0][1].axis('off')

    ax[1][0].imshow(opened, cmap='gray')
    ax[1][0].set_title('Opened')
    ax[1][0].axis('off')

    ax[1][1].imshow(closed, cmap='gray')
    ax[1][1].set_title('Closed')
    ax[1][1].axis('off')

    plt.show()

    return dilation, erosion, opened, closed

# - ⁠Automatic Thresholding (Segmentation)
def otsu_segmentation(image):

    T = filters.threshold_otsu(image)
    binary = image >= T

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].hist(image.ravel(), bins=256)
    ax[1].axvline(T, color='red')
    ax[1].set_title(f"Histogram (T = {T:.2f})")

    ax[2].imshow(binary, cmap="gray")
    ax[2].set_title("Binary Image")
    ax[2].axis("off")

    plt.show()

    return binary

# - ⁠ Dithering

def floyd_steinberg_dithering(image_path):

    image = Image.open(image_path).convert('L')

    # Apply Floyd-Steinberg dithering (1-bit)
    dithered = image.convert('1', dither=Image.FLOYDSTEINBERG)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original (8 bits)")
    ax[0].axis('off')

    ax[1].imshow(dithered, cmap='binary')
    ax[1].set_title("Floyd-Steinberg Dither (1 bit)")
    ax[1].axis('off')

    plt.show()

    return dithered