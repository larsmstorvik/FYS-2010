import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def question1():
    sinogram = np.load("pre_data/sinogram.npy")
    # Plot the sinogram
    plt.figure(figsize=(10, 8))
    plt.imshow(sinogram, cmap='gray', aspect='auto')
    plt.colorbar(label="Intensity")
    plt.xlabel("Detector Position")
    plt.ylabel("Projection Angle")
    plt.title("Sinogram")
    plt.savefig("data/task5/sinogram.png")


def inverse_radon(sinogram, theta, image_size=None, filter_type='ramp', cutoff=1.0, interpolation_type='linear', normalization_factor=1.0):
    """
    Inverse Radon Transform for non-square sinograms.
    
    Args:
        sinogram (np.ndarray): 2D array of shape (num_angles, num_detectors)
        theta (np.ndarray): 1D array of projection angles in degrees
        image_size (tuple): (height, width) of output image. Auto-detected if None.
        filter_type (str): Filter type ('ramp', 'hann' or None)
        cutoff (float): Frequency cutoff (0-1)
        
    Returns:
        np.ndarray: Reconstructed image
    """
    num_angles, num_detectors = sinogram.shape
    
    # Auto-detect image size if not provided
    if image_size is None:
        diag = int(np.ceil(num_detectors / np.sqrt(2)))
        image_size = (diag, diag)
    
    # 1. Apply frequency domain filtering
    filtered_sino = np.zeros_like(sinogram)
    for i in range(num_angles):
        filtered_sino[i, :] = apply_filter(sinogram[i, :], filter_type, cutoff)
    
    # 2. Backprojection
    recon = np.zeros(image_size)
    center = np.array(image_size) // 2
    y, x = np.indices(image_size)
    x = x - center[1]
    y = y - center[0]
    
    for angle_idx, angle in enumerate(theta):
        theta_rad = np.deg2rad(angle)
        
        # Calculate detector positions for this angle
        detector_pos = x * np.cos(theta_rad) + y * np.sin(theta_rad)
        
        # Map to detector indices
        detector_idx = (detector_pos + num_detectors/2) * (num_detectors-1)/num_detectors
        
        # Create interpolation function for this projection
        interp_fn = interp1d(np.arange(num_detectors), 
                            filtered_sino[angle_idx, :],
                            kind=interpolation_type,
                            bounds_error=False,
                            fill_value=0)
        
        # Interpolate and accumulate
        recon += interp_fn(detector_idx)

    # Normalization
    return recon * np.pi / (2 * num_angles) * normalization_factor

# Helper function from previous implementation
def apply_filter(projection, filter_type='ramp', cutoff=1.0):
    n = len(projection)
    fourier = np.fft.fft(projection)
    freq = np.fft.fftfreq(n)
    
    if filter_type == 'ramp':
        filt = np.abs(freq)
    elif filter_type == 'hann':
        filt = np.abs(freq) * (0.5 + 0.5 * np.cos(np.pi * freq / cutoff))
    elif filter_type == None:
        filt = np.ones(len(freq))
    else:
        raise ValueError(f"Unknown filter: {filter_type}")

    filt[freq > cutoff] = 0
    filt[freq < -cutoff] = 0
    
    return np.fft.ifft(fourier * filt).real

def question2():
    sinogram = np.load("pre_data/sinogram.npy")
    theta = np.linspace(0., 180., sinogram.shape[0], endpoint=False)  # Angles from 0 to 180 degrees
    print("Number of angles:",len(theta))

    # Reconstruct the image using the inverse Radon transform
    reconstructed_image_hann = inverse_radon(sinogram, theta, filter_type="hann")
    reconstructed_image_ramp = inverse_radon(sinogram, theta, filter_type="ramp")
    # Plot the results
    plt.figure(figsize=(16, 10))

    # Reconstructed image using ramp filter
    plt.subplot(1, 2, 1)
    plt.title("Reconstructed Image (Ramp Filter)")
    plt.imshow(reconstructed_image_ramp, cmap='gray')
    plt.axis('off')

    # Reconstructed image using hann filter
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image (Hann Filter)")
    plt.imshow(reconstructed_image_hann, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("data/task5/sinogram_reconstructed_me.png")


def question3():
    sinogram = np.load("pre_data/sinogram.npy")
    
    # Downsample the sinogram by a factor of 4
    sinogram_fast = sinogram[::4,:].copy()
    theta = np.linspace(0., 180., sinogram_fast.shape[0], endpoint=False)  # Angles from 0 to 180 degrees
    print("Number of angles:",len(theta))

    # Reconstruct the image using the inverse Radon transform
    reconstructed_image = inverse_radon(sinogram_fast, theta, filter_type="hann")

    # Plotting the reconstructed image
    plt.figure(figsize=(12, 4))
    # Reconstructed image
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed_image, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("data/task5/sinogram_fast_reconstructed_me.png")

def experiment_cutoff(sinogram, theta):

    reconstructed_image_1 = inverse_radon(sinogram, theta, filter_type="hann", cutoff=1.0)
    reconstructed_image_0_8 = inverse_radon(sinogram, theta, filter_type="hann", cutoff=0.8)
    reconstructed_image_0_5 = inverse_radon(sinogram, theta, filter_type="hann", cutoff=0.5)
    reconstructed_image_0_3 = inverse_radon(sinogram, theta, filter_type="hann", cutoff=0.3)
    
    # Plot the results
    plt.figure(figsize=(16, 10))

    # Reconstructed image using ramp filter
    plt.subplot(2, 2, 1)
    plt.title("Reconstructed Image: cutoff = 1.0")
    plt.imshow(reconstructed_image_1, cmap='gray')
    plt.axis('off')

    # Reconstructed image using hann filter
    plt.subplot(2, 2, 2)
    plt.title("Reconstructed Image: cutoff = 0.8")
    plt.imshow(reconstructed_image_0_8, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Reconstructed Image: cutoff = 0.5")
    plt.imshow(reconstructed_image_0_5, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Reconstructed Image: cutoff = 0.3")
    plt.imshow(reconstructed_image_0_3, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("data/task5/experiments/hann__cutoff_1_08_05_03.png")

def experiment_interpolation_type(sinogram, theta):
    reconstructed_image_linear = inverse_radon(sinogram, theta, filter_type="hann", interpolation_type='linear')
    reconstructed_image_cubic = inverse_radon(sinogram, theta, filter_type="hann", interpolation_type='cubic')
    reconstructed_image_nearest = inverse_radon(sinogram, theta, filter_type="hann", interpolation_type='nearest')


    plt.figure(figsize=(16, 10))

    # Reconstructed image using hann filter
    plt.subplot(2, 2, 1)
    plt.title("Reconstructed Image: interpolation type = linear")
    plt.imshow(reconstructed_image_linear, cmap='gray')
    plt.axis('off')

    # Reconstructed image using hann filter
    plt.subplot(2, 2, 2)
    plt.title("Reconstructed Image: interpolation type = cubic")
    plt.imshow(reconstructed_image_cubic, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Reconstructed Image: interpolation type = nearest")
    plt.imshow(reconstructed_image_nearest, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("data/task5/experiments/interpol_type_linear_cubic_nesrest_hann.png")

def experiment_normalization_factor(sinogram, theta):
    reconstructed_image_08 = inverse_radon(sinogram, theta, filter_type="ramp", normalization_factor=0.5)
    reconstructed_image_10 = inverse_radon(sinogram, theta, filter_type="ramp", normalization_factor=1.0)
    reconstructed_image_12 = inverse_radon(sinogram, theta, filter_type="ramp", normalization_factor=1.5)


    plt.figure(figsize=(16, 10))

    # Reconstructed image using ramp filter
    plt.subplot(2, 2, 1)
    plt.title("Reconstructed Image: normalization factor = 0.8")
    plt.imshow(reconstructed_image_08, cmap='gray')
    plt.axis('off')

    # Reconstructed image using ramp filter
    plt.subplot(2, 2, 2)
    plt.title("Reconstructed Image: normalization factor = 1.0")
    plt.imshow(reconstructed_image_10, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Reconstructed Image: normalization factor = 1.2")
    plt.imshow(reconstructed_image_12, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("data/task5/experiments/normalization_factors_08_10_12_ramp.png")


def question3_experiment():
    """
    Experiments with these parameters:
    - Filter type: Hann, ramp
    - Cutoff: 0.5, 0.8, 1.0
    - Interpolation type: Linear, cubic, 'nearest'
    - Normalization factor: 0.8, 1.0, 1.2
    """
    sinogram = np.load("pre_data/sinogram.npy")
    
    # Downsample the sinogram by a factor of 4
    sinogram_fast = sinogram[::4,:].copy()
    theta = np.linspace(0., 180., sinogram_fast.shape[0], endpoint=False)  # Angles from 0 to 180 degrees
    print("Number of angles:",len(theta))

    # Experiment with cutoff sizes
    #experiment_cutoff(sinogram_fast, theta)

    # Experiment with interpolation types
    #experiment_interpolation_type(sinogram_fast, theta)

    # Experiment with normalization factors
    experiment_normalization_factor(sinogram_fast, theta)
if __name__ == '__main__':
    question1()
    question2()
    question3()
    question3_experiment()

