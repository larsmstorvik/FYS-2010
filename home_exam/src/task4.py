import pywt
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np



def question1():
    # Load image
    image = io.imread("pre_data/IMG_MRA2.png")  # Replace with actual path
    # Check if the image has an alpha channel and remove it
    if image.shape[-1] == 4:  # RGBA image
        image = image[:, :, :3]  # Keep only the RGB channels
    gray_image = color.rgb2gray(image)  # Convert to grayscale
    # Perform 3-level DWT decomposition
    coeffs = pywt.wavedec2(gray_image, wavelet='haar', level=3)
    cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

    # Plot results
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes[0, 0].imshow(cA3, cmap='gray')
    axes[0, 0].set_title('Approximation Coefficients (LL3)')

    axes[0, 1].imshow(cH3, cmap='gray')
    axes[0, 1].set_title('Horizontal Detail (LH3)')

    axes[0, 2].imshow(cV3, cmap='gray')
    axes[0, 2].set_title('Vertical Detail (HL3)')

    axes[0, 3].imshow(cD3, cmap='gray')
    axes[0, 3].set_title('Diagonal Detail (HH3)')

    axes[1, 0].imshow(cH2, cmap='gray')
    axes[1, 0].set_title('Horizontal Detail (LH2)')

    axes[1, 1].imshow(cV2, cmap='gray')
    axes[1, 1].set_title('Vertical Detail (HL2)')

    axes[1, 2].imshow(cD2, cmap='gray')
    axes[1, 2].set_title('Diagonal Detail (HH2)')

    axes[1, 3].imshow(cD1, cmap='gray')
    axes[1, 3].set_title('Diagonal Detail (HH1)')

    plt.tight_layout()
    plt.savefig("data/task4/test/IMG_MRA1.png")

def question2():
    # Load image
    image = io.imread("pre_data/IMG_MRA1.png")  # Replace with actual path
    # Get image dimensions
    M, N, G = image.shape
    print(f"Image size: {M} x {N}")

    # Compute maximum decomposition levels
    max_levels = np.log2(min(M, N))
    #max_levels = int(max_levels)
    print(f"Maximum decomposition levels: {max_levels}")

def question3():
    # Load image
    image = io.imread("pre_data/IMG_MRA1.png", as_gray=True)

    # Perform 3-level, 5-level, and 8-level DWT
    levels = [3, 5, 8]

    # Select a detail coefficient
    chosen_coeff = np.random.choice(['cH', 'cV', 'cD'])  # Randomly select
    
    for level in levels:
        # Perform discrete wavelet decomposition
        coeffs = pywt.wavedec2(image.copy(), 'haar', level=level)
        cA, detail_coeffs = coeffs[0], coeffs[1:]  # Approximate and detailed
        
        print(f"Modifying {chosen_coeff} at lowest level (level {level})")

        # Choose coefficients of the lovest level (level 3, 5, and 8 respectfyully)
        cH, cV, cD = detail_coeffs[0]

        # Modify all values of the one coefficient and set all values to zero
        if chosen_coeff == 'cH':
            cH = cH * 0  
        elif chosen_coeff == 'cV':
            cV = cV * 0
        elif chosen_coeff == 'cD':
            cD = cD * 0
        detail_coeffs[0] = (cH, cV, cD)  # Ensure reassigning as tuple

        coeffs_modified = [cA] + detail_coeffs

        # Reconstruct the image
        reconstructed = pywt.waverec2(coeffs_modified, 'haar')

        # Plot the reconstructed image
        plt.figure(figsize=(5, 5))
        plt.imshow(reconstructed, cmap='gray')
        plt.title(f'Reconstructed Image - Level {level}. Chosen coeff: {chosen_coeff}')
        plt.axis('off')
        img_path = f"data/task4/reconstructed_image_level_{level}.png"
        plt.savefig(img_path)
        plt.close()
    # Plot original image
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.title(f'Original image')
    plt.axis('off')
    img_path = "data/task4/original_image_Q3.png"
    plt.savefig(img_path)
    plt.close()

if __name__ == "__main__":
    #question1()
    #question2()
    question3()