import cv2
import numpy as np
import matplotlib.pyplot as plt



# Plot original image
def plot_image(image, title, path):
    """
    Plots an image using matplotlib and saves it to a specified path.
    Parameters:
        image (ndarray): The image data to be plotted.
        title (str): The title of the plot.
        path (str): The path where the image will be saved, appended with 'data/task1/original/org_' and '.png'.
    Returns:
        None
    """
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.xlabel("")
    text = "data/task1/original/org_" + path + ".png"
    plt.savefig(text)

# Function to plot histogram
def plot_histogram(image, title, path):
    """
    Plots a histogram of the pixel intensities of an image and saves it to a file.
    Parameters:
        image (numpy.ndarray): The input image for which the histogram is to be plotted.
        title (str): The title of the histogram plot.
        path (str): The path where the histogram image will be saved, appended with 'data/task1/hist/hist_' and '.png'.
    Returns:
        None
    """
    plt.figure()
    plt.hist(image.ravel(), bins=256, density=False, histtype='step', color='black')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    text = "data/task1/hist/hist_" + path + ".png"
    plt.savefig(text)

def plot_homogenous_region(image, title, path, x, y, w, h):
    """
    Plots a region of an image and its histogram.
    This function creates a figure with two subplots: one displaying the image with a marked rectangular region,
    and another displaying the histogram of pixel intensities within that marked region. The plot is then saved
    to a specified path.
    Parameters:
        image (ndarray): The input image as a 2D numpy array.
        title (str): The title for the image plot.
        path (str): The base path for saving the output plot.
        x (int): The x-coordinate of the top-left corner of the rectangular region.
        y (int): The y-coordinate of the top-left corner of the rectangular region.
        w (int): The width of the rectangular region.
        h (int): The height of the rectangular region.
    Returns:
        None
    """
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
    text = "data/task1/homogen_hist/homogenous_hist_" + path + "_" + str(x) + "_" + str(y) + "_" + str(w) + "_" + str(h) + ".png"
    plt.savefig(text)


def pad_to_next_power_of_two(image):
    """
    Pads the given image to the next power of two dimensions.
    This function takes an image (2D numpy array) and pads it with zeros 
    so that both its height and width are the next power of two.
    Parameters:
        image (numpy.ndarray): A 2D array representing the image to be padded.
    Returns:
        numpy.ndarray: A new 2D array with dimensions padded to the next power of two.
    """

    # Get the dimensions of the image
    height, width = image.shape
    
    # Find the next power of two for both dimensions
    new_height = 2**np.ceil(np.log2(height)).astype(int)
    new_width = 2**np.ceil(np.log2(width)).astype(int)
    
    # Pad the image with zeros to the new dimensions
    padded_image = np.pad(image, ((0, new_height - height), (0, new_width - width)), mode='constant', constant_values=0)
    
    return padded_image


def fft2d(image):
    """
    Compute the 2-dimensional Fast Fourier Transform (FFT) of an image and shift the zero frequency component to the center of the spectrum.
    Parameters:
        image (numpy.ndarray): A 2D array representing the input image.
    Returns:
        numpy.ndarray: The shifted 2D FFT of the input image.
    """
    return np.fft.fftshift(np.fft.fft2(image))


def plot_fft(image, fft_result, title, path):
    """
    Plots the original image and its 2D FFT magnitude spectrum, and saves the plot as a PNG file.
    Parameters:
        image (ndarray): The original image to be plotted.
        fft_result (ndarray): The 2D FFT result of the image.
        title (str): The title to be used for the plots.
        path (str): The path where the plot image will be saved (excluding file extension).
    Returns:
        None
    """

    # Plot original image and its 2D FFT magnitude spectrum

    # Plot the original image
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Original Image: {title}")
    plt.axis('off')

    # Plot the magnitude spectrum of the FFT (log scale for visibility)
    magnitude_spectrum = np.log(np.abs(fft_result) + 1)  # log scale for better visibility
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(f"Magnitude Spectrum (2D FFT): {title}")
    plt.axis('off')

    plt.tight_layout()
    text = "data/task1/fft/fft_" + path + ".png"
    plt.savefig(text)

if __name__ == "__main__":
    images = ["pre_data/cameramanA.png", "pre_data/cameramanB.png", "pre_data/cameramanC.png",
              "pre_data/cameramanD.png", "pre_data/cameramanE.png", "pre_data/cameramanF.png"]

    for img_path in images:
        # Read pictures
        img = cv2.imread(img_path)
        #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_path_org = img_path.split("/")[1].split(".")[0]

        """ VISUAL INSPECTION """
        # Plot Homogenous region
        plot_image(img, f"Original Image {img_path_org}", img_path_org)

        """ HISTOGRAM ANALYSIS """
        # Plot histogram
        plot_histogram(img, f"Histogram of {img_path_org}", img_path_org)

        # Plot homogenous region
        plot_homogenous_region(img, img_path_org, img_path_org, x=180, y=0, w=75, h=100)
        plot_homogenous_region(img, img_path_org, img_path_org, x=45, y=150, w=35, h=30)

        """ FOURIER TRANSFORM ANALYSIS"""
        # # Pad the image to the next power of two
        # padded_image = pad_to_next_power_of_two(img)
        # # Apply the 2D FFT to the padded image
        # padded_fft_result = fft2d(padded_image)
        # # Plot the results
        # plot_fft(img, padded_fft_result, f"padded {img_path_org}", f"{img_path_org}_padded")
