import numpy as np
import matplotlib.pyplot as plt
import cv2


# Plot image
def plot_image(image, title, path):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)

    text = "data/task3/test/" + path + ".png"
    plt.savefig(text)


if __name__ == "__main__":
    robot_img = cv2.imread("pre_data/robot.jpg")  # Replace with your image path
    pig_img = cv2.imread("pre_data/pig.jpg")  # Replace with your image path

    # Turn both images into grayscale
    robot_gray = cv2.cvtColor(robot_img, cv2.COLOR_BGR2GRAY)
    pig_gray = cv2.cvtColor(pig_img, cv2.COLOR_BGR2GRAY)

    # Rezise robot.jpg to the same size as pig.jpg
    robot_resized = cv2.resize(robot_gray, (pig_gray.shape[1], pig_gray.shape[0]))

    # Compute the Fourier Transform of both images
    F_robot = np.fft.fft2(robot_resized)
    F_pig = np.fft.fft2(pig_gray)
    
    # Compute the magnitude and phase of the Fourier Transform of both images
    mag_robot, phase_robot = np.abs(F_robot), np.angle(F_robot)
    mag_pig, phase_pig = np.abs(F_pig), np.angle(F_pig)

    # swap the phase of the robot image with the pig image
    F_robot_swapped = mag_robot * np.exp(1j * phase_pig)
    F_pig_swapped = mag_pig * np.exp(1j * phase_robot)

    # Compute the inverse Fourier Transform of the swapped images
    robot_swapped = np.fft.ifft2(F_robot_swapped)
    pig_swapped = np.fft.ifft2(F_pig_swapped)

    # display the swapped images
    plot_image(np.abs(robot_swapped), "Magnitude: robot, phase: pig", "swapped_robot")
    plot_image(np.abs(pig_swapped), "Magnitude: pig, phase: robot", "swapped_pig")
    plt.savefig("data/task3/test/swapped_pig.png")