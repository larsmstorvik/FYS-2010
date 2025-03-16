import numpy as np
import matplotlib.pyplot as plt
import cv2




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
    print(image)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    text = "data/task2/hist/hist_" + path + ".png"
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
    text = "data/task2/homogenous/hist_" + path + "_" + str(x) + "_" + str(y) + "_" + str(w) + "_" + str(h) + ".png"
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


def plot_fft(image, fft_result, title, path):
    # Plot original image and its 2D FFT magnitude spectrum

    # Plot the original image
    plt.figure(figsize=(12, 6))

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
    text = "data/task2/" + path + "_fft.png"
    plt.savefig(text)


def plot_homogenous_hist_and_fft(image, image_path, x, y, w, h):
    # Plot the histogram of the original image
    plot_homogenous_hist(image, "Noisy Liver", img_path_org, x, y, w, h)
    
    # Plot 2D FFT of homogeneous region
    cropped_image = image[y:y+h, x:x+w]
    # Pad the image to the next power of two
    padded_image = pad_to_next_power_of_two(cropped_image)
    cropped_fft_result = fft2d(padded_image)
    plot_fft(padded_image, cropped_fft_result,"Noisy Liver (Homogeneous Region)",  "homogenous/" + img_path_org + "_homogeneous_region"+ "_" + str(x) + "_" + str(y) + "_" + str(w) + "_" + str(h))




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



if __name__ == "__main__":
    image_path = "pre_data/LiverNoisy.png"
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_path_org = image_path.split("/")[1].split(".")[0]

    """ Original image"""
    # Plot the original image
    #plot_image(image, f"Original Noisy Liver", img_path_org)
    
    """ Histogram """
    # Plot the histogram of the original image
    #plot_histogram(image, "Histogram of Noisy Liver", img_path_org)

    """ 2D FFT """
    # pad the image to the next power of two
    padded_image = pad_to_next_power_of_two(image)
    # Perform 2D FFT
    fft_result = fft2d(padded_image)
    # Plot 2D FFT
    plot_fft(padded_image, fft_result, "Noisy Liver", img_path_org + "_padded")


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
    notch_filter(image, img_path_org)
