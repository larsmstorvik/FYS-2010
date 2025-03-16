import cv2
import numpy as np
import matplotlib.pyplot as plt



# Plot original image
def plot_image(image, title, path):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)

    text = "data/task1/test/org_" + path + ".png"
    plt.savefig(text)

# Function to plot histogram
def plot_histogram(image, title, path):
    plt.figure()
    plt.hist(image.ravel(), bins=256, density=False, histtype='step', color='black')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    text = "data/task1/hist/hist_" + path + ".png"
    plt.savefig(text)

# Function to plot homogenous region
def plot_homogenous_region(image, title, path, x, y, w, h):
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
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_path_org = img_path.split("/")[1].split(".")[0]

        # Plot original image
        #plot_image(img, f"Original Image {img_path_org}", img_path_org)

        # Plot histogram
        #plot_histogram(img, f"Histogram of {img_path_org}", img_path_org)

        # Plot homogenous region
        #plot_homogenous_region(img, img_path_org, img_path_org, x=180, y=0, w=75, h=100)
        #plot_homogenous_region(img, img_path_org, img_path_org, x=45, y=150, w=35, h=30)

        # Pad the image to the next power of two
        padded_image = pad_to_next_power_of_two(img)
        # Apply the 2D FFT to the padded image
        padded_fft_result = fft2d(padded_image)
        # Plot the results
        plot_fft(img, padded_fft_result, f"padded {img_path_org}", f"{img_path_org}_padded")


        # Perform 2D FFT
        fft_result = fft2d(img)
        # Plot 2D FFT
        plot_fft(img, fft_result, img_path_org, img_path_org)

        print(f"Image {img_path_org} processed.")