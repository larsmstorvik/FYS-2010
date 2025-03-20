import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

# code for the inverse radon transform

def inverse_radon(sinogram, theta, image_size=None, filter_type='ramp', cutoff=1.0):
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
                            kind='linear',
                            bounds_error=False,
                            fill_value=0)
        
        # Interpolate and accumulate
        recon += interp_fn(detector_idx)

    # Normalization
    return recon * np.pi / (2 * num_angles)

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



sinogram = np.load("sinogram.npy")
theta = np.linspace(0., 180., sinogram.shape[0], endpoint=False)  # Angles from 0 to 180 degrees
print("Number of angles:",len(theta))

# Reconstruct the image using the inverse Radon transform
reconstructed_image = inverse_radon(sinogram, theta, filter_type=None)

# Plotting the reconstructed image
plt.figure(figsize=(12, 4))
# Reconstructed image
plt.title("Reconstructed Image")
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig("../data/task5/sinogram_reconstructed_by_CT_code.png")
