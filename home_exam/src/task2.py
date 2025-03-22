import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import median_filter


# Function to plot histogram
def plot_histogram(image, title, path):
    plt.figure()
    plt.hist(image.ravel(), bins=256, density=False, histtype='step', color='black')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    text = "data/task2/" + path + ".png"
    plt.savefig(text)
    plt.close()

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
    plt.close()

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

    if image is not None and image.any():
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
    else:
        # Only plot the fft
        magnitude_spectrum = np.log(np.abs(fft_result) + 1)  # log scale for better visibility
        plt.imshow(magnitude_spectrum, cmap='plasma', extent=(-fft_result.shape[1]//2, fft_result.shape[1]//2, -fft_result.shape[0]//2, fft_result.shape[0]//2))
        plt.title(f"Frequency Spectrum: {title}")
        plt.xlabel("Frequency (u)")
        plt.ylabel("Frequency (v)")
        

    plt.tight_layout()
    text = "data/task2/" + path + ".png"
    plt.savefig(text)
    plt.close()


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
    plt.close()

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
    plt.subplot(2, 3, 1)
    plt.imshow(original_spectrum, cmap='plasma')
    plt.title("Original spectrum")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(notch_mask, cmap='gray')
    plt.title("Notch Filter Mask")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(np.log(np.abs(fft_filtered_spectrum) + 1), cmap='plasma')
    plt.title("Added Notch Filter")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(image, cmap='Reds')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(fft_inverse_filtered_spectrum, cmap='Reds')
    plt.title("Notch Filter Mask rejects this")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(filtered_image, cmap='Reds')
    plt.title("Filtered Image")
    plt.axis('off')

    plt.savefig("data/task2/RBG/R/"+path+".png")
    plt.close()

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

def plot_denoised_green_chanel(image, filtered_image, path):
    # Show image and the hist
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

    plt.savefig("data/task2/RBG/G/"+path+".png")
    plt.close()


def denoise_green_channel(image):

    kernel_size = (3, 3)
    # when sigma is 0, GaussianBlur wil automatically select an appropriate sigma
    sigma = 0
    filtered_image = cv2.GaussianBlur(image, kernel_size, sigma)
 
    plot_denoised_green_chanel(image, filtered_image, "denoised_arithmetic_mean")
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
    plt.close()

def denoise_blue_channel(image):
    """
    Denoise the salt and pepper noise in the blue channel
    """

    # Apply median filter to the image
    denoised_image_3 = median_filter(image, size=3)
    
    plot_denoised_blue_chanel(image, denoised_image_3, "denoised_blue_channel_size_3")

    return denoised_image_3



def plot_combined_denoised_image(image_denoised):
    # Plot the combined image
    plt.figure()
    plt.imshow(image_denoised, cmap='gray')
    plt.title("Combined Denoised Image")    
    plt.axis('off')
    plt.savefig("data/task2/RBG/combined_denoised_image.png")
    plt.close()

def combine_channels(R, G, B):
    # Combine the denoised channels
    image_denoised = cv2.merge((R, G, B))
    
    # convert to grayscale
    image_denoised = cv2.cvtColor(image_denoised, cv2.COLOR_BGR2GRAY)

    # plot the combined image
    plot_combined_denoised_image(image_denoised)

    return image_denoised


def high_boost_filter(image, sigma=2, weight=1.0):
    """
    Apply a high-boost filter to an image.
    Parameters:
        image (numpy.ndarray): Input image to be filtered.
        sigma (float, optional): Standard deviation for Gaussian kernel. Default is 2.
        weight (float, optional): Weight for the high-boost filter. Default is 1.0.
    Returns:
        numpy.ndarray: The filtered image with the same dimensions as the input image.
    """

    f_LP = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    g = cv2.addWeighted(image, 1 + weight, f_LP, -weight, 0)
    return np.clip(g, 0, 255).astype(np.uint8)


""" QUESTION 5 """
def spacial_domain_enhancement(image):
    """
    Enhances the given image using a high-boost filter and plots the original and filtered images.
    Parameters:
        image (ndarray): The input image to be enhanced.
        Returns:
    None
    """
    

    g = high_boost_filter(image)


    # Plot the original and filtered image
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original image")

    plt.subplot(1, 2, 2)
    plt.imshow(g, cmap='gray')
    plt.title("Filtered image")
    plt.savefig("data/task2/enhanced_image/spacial_domain_enhancement_final.png")
    plt.close()


def fourier_domain_enhancement(image):
    """
    Enhance an image using Fourier domain techniques.
    This function applies a high-pass filter in the Fourier domain to enhance the high-frequency components of the input image. 
    The enhanced image is then combined with the original image to produce the final enhanced image.
    Parameters:
        image (numpy.ndarray): The input grayscale image to be enhanced.
    Returns:
        None
    """
    
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    cutoff = 5
    low_pass_filter = gaussian_low_pass_filter(image.shape, cutoff)

    high_pass_filter = 1 - low_pass_filter

    enhanced_spectrum = fshift * high_pass_filter
    enhanced_image_high_pass = np.abs(np.fft.ifft2(np.fft.ifftshift(enhanced_spectrum)))

    weight = 0.5
    enhanced_image_final = image + weight * enhanced_image_high_pass
    enhanced_image_final = np.clip(enhanced_image_final, 0, 255)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1), plt.imshow(np.log(1 + np.abs(high_pass_filter)), cmap='gray'), plt.title("Fourier Spectrum")
    plt.subplot(2, 2, 2), plt.imshow(image, cmap='gray'), plt.title("Original Image")
    plt.subplot(2, 2, 3), plt.imshow(np.log(1 + np.abs(enhanced_spectrum)), cmap='gray'), plt.title("High-Pass Filtered")
    plt.subplot(2, 2, 4), plt.imshow(enhanced_image_high_pass, cmap='gray'), plt.title("Gaussian High-Pass Filtered")
    plt.savefig("data/task2/enhanced_image/fourier_domain_enhancement.png")
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.imshow(enhanced_image_final, cmap='gray')
    plt.title(f"original image + enganceh image. weight = {weight}")
    plt.savefig("data/task2/enhanced_image/fourier_domain_enhancement_final.png")
    plt.close()


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
    plot_RBG_channels_unmodified(R, B, G)
    

    for channel, name, identifier in zip([R, G, B], ["Red Channel", "Green Channel", "Blue Channel"],  ["R", "G", "B"]):
        """ Plot histograms of chanels """
        plot_histogram(channel, f"Histogram of {name}", f"RBG/{identifier}/histogram")
        
        """ Plot 2D FFT of chanels """
        # Perform 2D FFT
        padded_image = pad_to_next_power_of_two(channel)
        fft_result = fft2d(padded_image)
        # Plot 2D FFT
        plot_fft(fft_result, f"2D FFT of {name}", f"RBG/{identifier}/fft")
    
        """ Plot homogenous histogram and fft """
        x, y, w, h = 0, 0, 64, 64
        plot_homogenous_hist(channel, f"Homogenous histogram of {name}", f"RBG/{identifier}/histogram_homogenous", x, y, w, h)
        # Plot 2D FFT of homogeneous region
        cropped_image = channel[y:y+h, x:x+w]
        # Pad the image to the next power of two
        padded_cropped_image = pad_to_next_power_of_two(cropped_image)
        cropped_fft_result = fft2d(padded_cropped_image)
        plot_fft(cropped_fft_result,f"cropped 2D FFT of {name}",  f"RBG/{identifier}/fft_cropped_{x}_{y}_{w}_{h}", image=padded_cropped_image)

        x, y, w, h = 50, 105, 100, 105
        plot_homogenous_hist(channel, f"Homogenous histogram of {name}", f"RBG/{identifier}/histogram_homogenous", x, y, w, h)
        # Plot 2D FFT of homogeneous region
        cropped_image = channel[y:y+h, x:x+w]
        # Pad the image to the next power of two
        padded_cropped_image = pad_to_next_power_of_two(cropped_image)
        cropped_fft_result = fft2d(padded_cropped_image)
        plot_fft(cropped_fft_result,f"cropped 2D FFT of {name}",  f"RBG/{identifier}/fft_cropped_{x}_{y}_{w}_{h}", image=padded_cropped_image)

    """ Denoice channels """
    R_denoised = denoice_red_channel(R)
    G_denoised = denoise_green_channel(G)
    B_denoised = denoise_blue_channel(B)

    """ Combine channels """
    combined_image = combine_channels(R_denoised, G_denoised, B_denoised)

    """ Enhance image """
    enhance_image(combined_image)


   

if __name__ == "__main__":
    image_path = "pre_data/LiverNoisy.png"

    rbg_test(image_path)