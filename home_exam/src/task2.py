import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import median_filter, generic_filter
from PIL import Image

# Plot original image
def plot_image(image, title, path):
    plt.figure()
    plt.imshow(image)
    plt.title(title)

    text = "data/task2/test/org_" + path + ".png"
    plt.savefig(text)

# Function to plot histogram
def plot_histogram(image, title, path):
    plt.figure()
    plt.hist(image.ravel(), bins=256, density=False, histtype='step', color='black')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    text = "data/task2/" + path + ".png"
    plt.savefig(text)

# Function to plot homogenous region
def plot_homogenous_hist(image, title, path, x, y, w, h):
    # Create a figure with two subplots (one for the image and one for the histogram)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # First subplot: Image with marked square
    axs[0].imshow(image, cmap='gray')
    axs[0].add_patch(plt.Rectangle((x, y), w, h, edgecolor='b', facecolor='none', linewidth=2))
    axs[0].set_title(f"Image: {title}")
    axs[0].axis('off')
    
    # Second subplot: Histogram of the marked area
    cropped_image = image[y:y+h, x:x+w]
    axs[1].hist(cropped_image.ravel(), bins=256, range=[0,256])
    axs[1].set_title(f"Histogram of Area: {title}")
    axs[1].set_xlabel("Pixel Intensity")
    axs[1].set_ylabel("Frequency")
    
    # Show the plot
    plt.tight_layout()
    text = "data/task2/" + path + "_" + str(x) + "_" + str(y) + "_" + str(w) + "_" + str(h) + ".png"
    plt.savefig(text)


def pad_to_next_power_of_two(image):
    # Get the dimensions of the image
    height, width = image.shape
    
    # Find the next power of two for both dimensions
    new_height = 2**np.ceil(np.log2(height)).astype(int)
    new_width = 2**np.ceil(np.log2(width)).astype(int)
    
    # Pad the image with zeros to the new dimensions
    padded_image = np.pad(image, ((0, new_height - height), (0, new_width - width)), mode='constant', constant_values=0)
    
    return padded_image



def fft2d(image):
    # Perform 2D FFT by first applying FFT to each row and then to each column
    return np.fft.fftshift(np.fft.fft2(image))


def plot_fft(fft_result, title, path, image=None,):
    # Plot original image and its 2D FFT magnitude spectrum

    # Plot the original image
    plt.figure(figsize=(12, 6))

    if image == None:
        # Only plot the fft
        magnitude_spectrum = np.log(np.abs(fft_result) + 1)  # log scale for better visibility
        plt.imshow(magnitude_spectrum, cmap='plasma', extent=(-fft_result.shape[1]//2, fft_result.shape[1]//2, -fft_result.shape[0]//2, fft_result.shape[0]//2))
        plt.title(f"Frequency Spectrum: {title}")
        plt.xlabel("Frequency (u)")
        plt.ylabel("Frequency (v)")
    else:
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Original Image: {title}")
        #plt.axis('off')

        # Plot the magnitude spectrum of the FFT (log scale for visibility)
        magnitude_spectrum = np.log(np.abs(fft_result) + 1)  # log scale for better visibility
        plt.subplot(1, 2, 2)
        plt.imshow(magnitude_spectrum, cmap='plasma')
        plt.title(f"Magnitude Spectrum (2D FFT): {title}")
        #plt.axis('off')

    plt.tight_layout()
    text = "data/task2/" + path + ".png"
    plt.savefig(text)


def plot_homogenous_hist_and_fft(image, image_path, x, y, w, h):
    # Plot the histogram of the original image
    plot_homogenous_hist(image, "Noisy Liver", img_path_org, x, y, w, h)
    
    # Plot 2D FFT of homogeneous region
    cropped_image = image[y:y+h, x:x+w]
    # Pad the image to the next power of two
    padded_image = pad_to_next_power_of_two(cropped_image)
    cropped_fft_result = fft2d(padded_image)
    plot_fft(cropped_fft_result,"Noisy Liver (Homogeneous Region)",  "homogenous/" + img_path_org + "_homogeneous_region"+ "_" + str(x) + "_" + str(y) + "_" + str(w) + "_" + str(h), image=padded_image)


def plot_gaussian_filter_added(image, sigmax, sigmay):
    # Apply Gaussian filter to the image
    filtered_image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=sigmax, sigmaY=sigmay)
    
    # Plot the original and filtered images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f"Image after Gaussian Filter (sigmax={sigmax}, sigmay={sigmay})")
    plt.axis('off')
    
    plt.tight_layout()

    text = "data/task2/gaussian_filter/gaussian_filter_" + img_path_org + "_" + str(sigmax) + "_" + str(sigmay) + ".png"
    plt.savefig(text)

def create_notch_filter(image, center, radius):
    # Create a mask with the same dimensions as the image
    mask = np.ones_like(image, dtype=np.float64)
    
    # Get the dimensions of the image
    height, width = image.shape
    
    # Calculate the center of the image
    center_x, center_y = center
    
    # Create a meshgrid with the coordinates of each pixel
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Calculate the distance of each pixel to the center
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create a circular mask with the specified radius
    mask[distance < radius] = 0
    
    return mask

def notch_filter(image, image_path):
    # apply padding to the image
    padded_image = pad_to_next_power_of_two(image)

    # 2D FFT
    fft_result = fft2d(padded_image)
    magnitude_spectrum = np.log(np.abs(fft_result) + 1)
    # create notch filter mask
    n_rows = padded_image.shape[0]
    n_cols = padded_image.shape[1]
    c_row, c_col = (n_cols // 2, n_rows // 2)

    notch_mask = np.ones((n_rows, n_cols), np.uint8)
    notch_width = 5  # Width of the horizontal strip to remove
    
    #notch_mask[c_row-notch_width:c_row+notch_width, :] = 0  # Set mask to 0 along the horizontal line
    #notch_mask[c_row-notch_width:c_row+notch_width, c_col - 10:c_col + 10] = 1  # Set mask to 1 in the midle

    sigma = 10
    
    # Generate Gaussian notch mask
    for i in range(n_rows):
        for j in range(n_cols):
            # Calculate distance from the center (c_row, c_col)
            dist = i - c_row
            # Apply Gaussian function to create a smooth notch filter
            notch_mask[i, j] =np.exp(-(dist**2) / (2 * sigma**2))

    # Apply the notch filter mask
    fft_result_filtered = fft_result * notch_mask
    # Inverse 2D FFT
    filtered_image_padded = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_result_filtered)))
    # Remove padding
    filtered_image = filtered_image_padded[:image.shape[0], :image.shape[1]]

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(magnitude_spectrum, cmap='plasma')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(notch_mask, cmap='gray')
    plt.title("Notch Filter Mask")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(fft_result_filtered, cmap='plasma')
    plt.title("Added Notch Filter")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(filtered_image, cmap='gray')
    plt.title("Filtered Image")
    plt.axis('off')

    plt.savefig("data/task2/notch_filter/notch_filter_mask.png")


def plot_RBG_channels_unmodified(R, B, G):
    # Display the separate channels
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(R, cmap="Reds")
    ax[0].set_title("Red Channel")

    ax[1].imshow(G, cmap="Greens")
    ax[1].set_title("Green Channel")

    ax[2].imshow(B, cmap="Blues")
    ax[2].set_title("Blue Channel")

    plt.savefig("data/task2/RBG/rgb_channels.png")

""" QUESTION 4 """
def plot_denoised_red_chanel(image, original_spectrum, notch_mask, fft_filtered_spectrum, fft_inverse_filtered_spectrum, filtered_image, path):
    """
    Plots the denoising process of the red channel of an image using various stages of the Fourier Transform.
    Parameters:
        image (ndarray): The original image.
        original_spectrum (ndarray): The Fourier Transform spectrum of the original image.
        notch_mask (ndarray): The mask used for the notch filter.
        fft_filtered_spectrum (ndarray): The Fourier Transform spectrum after applying the notch filter.
        fft_inverse_filtered_spectrum (ndarray): The inverse Fourier Transform of the filtered spectrum.
        filtered_image (ndarray): The final filtered image.
        path (str): The path where the plot will be saved.
    Returns:
        None
    """
    
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 3, 1)
    plt.imshow(original_spectrum, cmap='plasma')
    plt.title("Original spectrum")
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(notch_mask, cmap='gray')
    plt.title("Notch Filter Mask")
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(np.log(np.abs(fft_filtered_spectrum) + 1), cmap='plasma')
    plt.title("Added Notch Filter")
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(image, cmap='Reds')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(fft_inverse_filtered_spectrum, cmap='Reds')
    plt.title("Notch Filter Mask rejects this")
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(filtered_image, cmap='Reds')
    plt.title("Filtered Image")
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.hist(image.ravel(), bins=256, density=False, histtype='step', color='black')
    plt.title("Histogram of Original Image")
    plt.xlabel("Pixel Intensity")

    plt.subplot(3, 3, 8)
    plt.hist(fft_inverse_filtered_spectrum.ravel(), bins=256, density=False, histtype='step', color='black')
    plt.title("Histogram of Filtered Image")
    plt.xlabel("Pixel Intensity")

    plt.subplot(3, 3, 9)
    plt.hist(filtered_image.ravel(), bins=256, density=False, histtype='step', color='black')
    plt.title("Histogram of Filtered Image")
    plt.xlabel("Pixel Intensity")

    plt.savefig("data/task2/RBG/R/"+path+".png")

def denoice_red_channel(image):
    """
    Denoise the red channel of an image using a notch filter in the frequency domain.
    Parameters:
        image (numpy.ndarray): Input image with the red channel to be denoised.
    Returns:
        numpy.ndarray: Denoised red channel image.
    The function performs the following steps:
        1. Pads the input image to the next power of two.
        2. Computes the 2D Fast Fourier Transform (FFT) of the padded image.
        3. Creates a Gaussian notch filter mask to remove horizontal noise.
        4. Applies the notch filter mask to the FFT result.
        5. Computes the inverse 2D FFT to obtain the filtered image.
        6. Removes the padding from the filtered image.
        7. Transforms the filtered image to uint8 format.
        8. Plots the original and filtered images for comparison.
    Note:
    - The function assumes that the input image is a 2D numpy array representing the red channel.
    - The function uses a Gaussian notch filter to remove horizontal noise in the frequency domain.
    - The function plots the original image, magnitude spectrum, notch filter mask, filtered FFT result, and the final denoised image.
    """

    # Apply padding
    padded_image = pad_to_next_power_of_two(image)
    # Perform 2D FFT
    fft_result = fft2d(padded_image)
    magnitude_spectrum = np.log(np.abs(fft_result) + 1)
    # Create notch filter mask
    n_rows = padded_image.shape[0]
    n_cols = padded_image.shape[1]
    c_row, c_col = (n_cols // 2, n_rows // 2)
    notch_mask = np.ones((n_rows, n_cols), np.uint8)
    notch_mask_inverse = np.zeros((n_rows, n_cols), np.uint8)

    sigma = 0.3
    # Generate Gaussian notch mask horizontally
    for i in range(n_rows):
        for j in range(n_cols):
            # Calculate distance from the center
            dist = i - c_row
            # Apply Gaussian function to create a smooth notch filter
            notch_mask[i, j] = 1 - np.exp(-(dist**2) / (2 * sigma**2))
            notch_mask_inverse[i, j] = 1 - np.exp(-(dist**2) / (2 * sigma**2))

    # Inverse the notch mask to get the inverse filter
    notch_mask_inverse = (1 - notch_mask_inverse)

    # Padd the center of the mask with ones
    notch_width = 10  # Width of the horizontal strip to remove
    notch_mask[c_col-notch_width:c_col+notch_width, c_row-notch_width:c_col+notch_width] = 1  # Set mask to 1 in the midle
    notch_mask_inverse[c_col-notch_width:c_col+notch_width, c_row-notch_width:c_col+notch_width] = 0  # Set mask to 0 in the midle

    # Apply the notch filter mask (and inverse mask)
    fft_result_filtered = fft_result * notch_mask
    fft_result_filtered_inverse = fft_result * notch_mask_inverse
    # Inverse 2D FFT
    filtered_image_padded = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_result_filtered)))
    filtered_image_padded_inverse = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_result_filtered_inverse)))
    # Remove padding
    filtered_image = filtered_image_padded[:image.shape[0], :image.shape[1]]
    filtered_image_inverse = filtered_image_padded_inverse[:image.shape[0], :image.shape[1]]

    # Transform filtered image to uint8 (this is necessary to combine the channels together at a later stage)
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    # Plot the original and filtered images
    plot_denoised_red_chanel(image, magnitude_spectrum, notch_mask, fft_result_filtered, filtered_image_inverse, filtered_image, "notch_filter_mask")

    return filtered_image


def gaussian_low_pass_filter(shape, cutoff):
    """
    Creates a Gaussian low-pass filter.
    Parameters:
        shape (tuple): The shape of the filter (P, Q).
        cutoff (float): The cutoff frequency for the filter.
    Returns:
        numpy.ndarray: A 2D array representing the Gaussian low-pass filter.
    """
    
    P, Q = shape
    u, v = np.meshgrid(np.arange(Q) - Q//2, np.arange(P) - P//2)
    D = np.sqrt(u**2 + v**2)
    H = np.exp(-(D**2) / (2 * (cutoff**2)))
    return H

def plot_denoised_green_chanel(image, original_spectrum, low_pass_filter, frequency_domain_added_filter, spatial_domain_added_inverse, filtered_image, path):
    # Show image and the hist

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 3, 1)
    plt.imshow(original_spectrum, cmap='plasma')
    plt.title("Original spectrum")
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(low_pass_filter, cmap='gray')
    plt.title("Gaussian Low Pass Filter")
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(np.log(np.abs(frequency_domain_added_filter) + 1), cmap='plasma')
    plt.title("Added Gaussian Low Pass Filter")
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(image, cmap='Greens')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(spatial_domain_added_inverse, cmap='Greens')
    plt.title("Gaussian Low Pass Filter rejects this")
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(filtered_image, cmap='Greens')
    plt.title("Filtered Image")
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.hist(image.ravel(), bins=256, density=False, histtype='step', color='black')
    plt.title("Histogram of Original Image")
    plt.xlabel("Pixel Intensity")

    plt.subplot(3, 3, 8)
    plt.hist(spatial_domain_added_inverse.ravel(), bins=256, density=False, histtype='step', color='black')
    plt.title("Histogram of Filtered Image")
    plt.xlabel("Pixel Intensity")

    plt.subplot(3, 3, 9)
    plt.hist(filtered_image.ravel(), bins=256, density=False, histtype='step', color='black')
    plt.title("Histogram of Filtered Image")
    plt.xlabel("Pixel Intensity")


    plt.savefig("data/task2/RBG/G/"+path+".png")


def denoise_green_channel(image):
    
    # from https://medium.com/@abhishekjainindore24/gaussian-noise-in-machine-learning-aab693a10170
    # denoised_img = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # # Apply padding
    # padded_image = pad_to_next_power_of_two(image)
    # # Perform 2D FFT
    # fft_result = fft2d(padded_image)
    # magnitude_spectrum = np.log(np.abs(fft_result) + 1)

    # cutoff = 10
    # # Create gaussian low pass filter
    # low_pass_filter = gaussian_low_pass_filter(fft_result.shape, cutoff)
    # high_pass_filter = 1 - low_pass_filter

    # # Apply the notch filter mask (and inverse mask)
    # fft_result_filtered = fft_result * low_pass_filter
    # fft_result_filtered_inverse = fft_result * high_pass_filter
    # # Inverse 2D FFT
    # filtered_image_padded = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_result_filtered)))
    # filtered_image_padded_inverse = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_result_filtered_inverse)))
    # # Remove padding
    # filtered_image = filtered_image_padded[:image.shape[0], :image.shape[1]]
    # filtered_image_inverse = filtered_image_padded_inverse[:image.shape[0], :image.shape[1]]

    # # Transform filtered image to uint8 (this is necessary to combine the channels together at a later stage)
    # filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    # # Show image and the hist
    # plot_denoised_green_chanel(image, magnitude_spectrum, low_pass_filter, fft_result_filtered, filtered_image_inverse, filtered_image, "denoised_gaussian_low_pass_filter")


    # Apply arithmetic mean filter:
    kernel_size = (5, 5)
    sigma = 0
    filtered_image = cv2.GaussianBlur(image, kernel_size, sigma)
 
    plt.figure(figsize=(12,8))

    plt.subplot(3, 2, 1)
    plt.imshow(image, cmap='Greens')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.imshow(filtered_image, cmap='Greens')
    plt.title("Filtered image")
    plt.axis('off')

    plt.subplot(3, 2, 3)
    plt.hist(image.ravel(), bins=256, density=False, histtype='step', color='black')
    plt.title("Histogram of Filtered Image")
    plt.xlabel("Pixel Intensity")

    plt.subplot(3, 2, 4)
    plt.hist(filtered_image.ravel(), bins=256, density=False, histtype='step', color='black')
    plt.title("Histogram of Filtered Image")
    plt.xlabel("Pixel Intensity")

    x, y, w, h = 50, 105, 100, 105  # Adjust these values as needed

    # Image with marked square
    plt.subplot(3, 2, 5)
    plt.imshow(filtered_image, cmap='gray')
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='b', facecolor='none', linewidth=2))
    plt.title("Marked Area in Original Image")
    plt.axis('off')

    cropped_image = filtered_image[y:y+h, x:x+w]
    # Histogram of the marked area
    plt.subplot(3, 2, 6)
    plt.hist(cropped_image.ravel(), bins=256, range=[0, 256], color='black', histtype='step')
    plt.title("Histogram of Marked Area")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.savefig("data/task2/RBG/G/"+"denoised_arithmetic_mean"+".png")

    return filtered_image


def plot_denoised_blue_chanel(image, denoised_img, path):
    # Plot the original and denoised images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='Blues')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(denoised_img, cmap='Blues')
    plt.title("Denoised Image size = 5")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.hist(image.ravel(), bins=256, density=False, histtype='step', color='black')
    plt.title("Histogram of Original Image")
    plt.xlabel("Pixel Intensity")

    plt.subplot(2, 2, 4)
    plt.hist(denoised_img.ravel(), bins=256, density=False, histtype='step', color='black')
    plt.title("Histogram of Denoised Image")
    plt.xlabel("Pixel Intensity")

    plt.tight_layout()
    plt.savefig("data/task2/RBG/B/"+path+".png")

def denoise_blue_channel(image):
    """
    Denoise the salt and pepper noise in the blue channel
    """

    # Apply median filter to the image
    denoised_image_3 = median_filter(image, size=3)
    denoised_image_5 = median_filter(image, size=5)
    denoised_image_7 = median_filter(image, size=7)
    
    #plot_denoised_blue_chanel(image, denoised_image_3, "denoised_blue_channel_size_3")

    return denoised_image_3

    

    # Plot denoised to size 3, 5, and 7
    """    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='Blues')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(denoised_image_3, cmap='Blues')
    plt.title("Denoised Image size = 3")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(denoised_image_5, cmap='Blues')
    plt.title("Denoised Image size = 5")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(denoised_image_7, cmap='Blues')
    plt.title("Denoised Image size = 7")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("data/task2/RBG/B/denoised_blue_channel.png")
    """


def plot_combined_denoised_image(image_denoised):
    # Plot the combined image
    plt.figure()
    plt.imshow(image_denoised, cmap='gray')
    plt.title("Combined Denoised Image")    
    plt.axis('off')
    plt.savefig("data/task2/RBG/combined_denoised_image.png")

def combine_channels(R, G, B):
    # Combine the denoised channels
    image_denoised = cv2.merge((R, G, B))
    
    # convert to grayscale
    image_denoised = cv2.cvtColor(image_denoised, cv2.COLOR_BGR2GRAY)

    # plot the combined image
    #plot_combined_denoised_image(image_denoised)

    return image_denoised



""" QUESTION 5 """
def spacial_domain_enhancement(image):
    """
    Using the laplacian filter to enhance the filter in spatial domain
    """
    # Spatial Domain Enhancement - Laplacian Filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=15)

    # Plot the original and filtered images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(laplacian, cmap='gray')
    plt.title("Laplacian Filtered Image")
    plt.axis('off')


    plt.savefig("data/task2/enhanced_image/spacial_domain_enhancement.png")



def fourier_domain_enhancement(image):
    """
    Using a high pass filter to enhance the image in the fourier domain
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)

    cutoff = 5
    low_pass_filter = gaussian_low_pass_filter(image.shape, cutoff)

    high_pass_filter = 1 - low_pass_filter

    enhanced_spectrum = fshift * high_pass_filter
    enhanced_image_high_pass = np.abs(np.fft.ifft2(np.fft.ifftshift(enhanced_spectrum)))
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1), plt.imshow(np.log(1 + np.abs(high_pass_filter)), cmap='gray'), plt.title("Fourier Spectrum")
    plt.subplot(2, 2, 2), plt.imshow(image, cmap='gray'), plt.title("Original Image")
    plt.subplot(2, 2, 3), plt.imshow(np.log(1 + np.abs(enhanced_spectrum)), cmap='gray'), plt.title("High-Pass Filtered")
    plt.subplot(2, 2, 4), plt.imshow(enhanced_image_high_pass, cmap='gray'), plt.title("Gaussian Low-Pass Filtered")
    plt.savefig("data/task2/enhanced_image/fourier_domain_enhancement.png")


def enhance_image(image):
    spacial_domain_enhancement(image)
    fourier_domain_enhancement(image)


def rbg_test(image_path):
    # Load the image
    image = cv2.imread(image_path)  # Replace with your image path
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Split the image into its RGB channels
    R, G, B = cv2.split(image)

    

    """ Plot all channels unmodified"""
    # plot_RBG_channels_unmodified(R, B, G)
    

    for channel, name, identifier in zip([R, G, B], ["Red Channel", "Green Channel", "Blue Channel"],  ["R", "G", "B"]):
        """ Plot histograms of chanels """
        # plot_histogram(channel, f"Histogram of {name}", f"RBG/{identifier}/histogram")
        
        """ Plot 2D FFT of chanels """
        # # Perform 2D FFT
        # padded_image = pad_to_next_power_of_two(channel)
        # fft_result = fft2d(padded_image)
        # # Plot 2D FFT
        # plot_fft(fft_result, f"2D FFT of {name}", f"RBG/{identifier}/fft")
    
        """ Plot homogenous histogram and fft """
        # x, y, w, h = 0, 0, 64, 64
        # plot_homogenous_hist(channel, f"Homogenous histogram of {name}", f"RBG/{identifier}/histogram_homogenous", x, y, w, h)
        # # Plot 2D FFT of homogeneous region
        # cropped_image = channel[y:y+h, x:x+w]
        # # Pad the image to the next power of two
        # padded_cropped_image = pad_to_next_power_of_two(cropped_image)
        # cropped_fft_result = fft2d(padded_cropped_image)
        # plot_fft(cropped_fft_result,f"cropped 2D FFT of {name}",  f"RBG/{identifier}/fft_cropped_{x}_{y}_{w}_{h}", image=padded_cropped_image)

        # x, y, w, h = 50, 105, 100, 105
        # plot_homogenous_hist(channel, f"Homogenous histogram of {name}", f"RBG/{identifier}/histogram_homogenous", x, y, w, h)
        # # Plot 2D FFT of homogeneous region
        # cropped_image = channel[y:y+h, x:x+w]
        # # Pad the image to the next power of two
        # padded_cropped_image = pad_to_next_power_of_two(cropped_image)
        # cropped_fft_result = fft2d(padded_cropped_image)
        # plot_fft(cropped_fft_result,f"cropped 2D FFT of {name}",  f"RBG/{identifier}/fft_cropped_{x}_{y}_{w}_{h}", image=padded_cropped_image)

    """ Denoice channels """
    R_denoised = denoice_red_channel(R)
    G_denoised = denoise_green_channel(G)
    B_denoised = denoise_blue_channel(B)

    """ Combine channels """
    combined_image = combine_channels(R_denoised, G_denoised, B_denoised)

    enhance_image(combined_image)


   

if __name__ == "__main__":
    image_path = "pre_data/LiverNoisy.png"
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_path_org = image_path.split("/")[1].split(".")[0]

    """ RBG test"""
    rbg_test(image_path)


    """ Original image"""
    # Plot the original image
    #plot_image(image, f"Original Noisy Liver", img_path_org)
    
    """ Histogram """
    # Plot the histogram of the original image
    #plot_histogram(image, "Histogram of Noisy Liver", "hist/hist_" + img_path_org)

    """ 2D FFT """
    # pad the image to the next power of two
    #padded_image = pad_to_next_power_of_two(image)
    # Perform 2D FFT
    #fft_result = fft2d(padded_image)
    # Plot 2D FFT
    #plot_fft(padded_image, fft_result, "Noisy Liver", img_path_org + "_padded")


    """ Homogenous region plot histogram and fft """

    x, y, w, h = 0, 0, 55, 55
    #plot_homogenous_hist_and_fft(image, img_path_org, x, y, w, h)

    #plot_homogenous_hist(image, "Noisy Liver", img_path_org, x, y, w, h)

    # Plot 2D FFT of homogeneous region
    #cropped_image = image[y:y+h, x:x+w]
    #cropped_fft_result = fft2d(cropped_image)
    #plot_fft(cropped_image, cropped_fft_result,"Noisy Liver (Homogeneous Region)",  "homogenous/" + img_path_org + "_homogeneous_region"+ "_" + str(x) + "_" + str(y) + "_" + str(w) + "_" + str(h))


    """ Gaussian filter """
    # Plot the original and filtered images
    #plot_gaussian_filter_added(image, sigmax=0, sigmay=0)
    #plot_gaussian_filter_added(image, sigmax=0.5,sigmay= 0.5)
    #plot_gaussian_filter_added(image, sigmax=1.0,sigmay= 1.0)
    #plot_gaussian_filter_added(image, sigmax=2.0,sigmay= 2.0)

    """ Notch filter """
    #notch_filter(image, img_path_org)
