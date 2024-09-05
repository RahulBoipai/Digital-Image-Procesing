import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from scipy.ndimage import convolve



def gaussian(x, sigma):
    return (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2) / (2 * sigma ** 2))

def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def bilateral_filter(image, d, sigmaColor, sigmaSpace):
    """ Apply bilateral filtering """
    # Initialize output array
    output = np.zeros_like(image)
    pad_width = d // 2
    padded_image = np.pad(image, pad_width, mode='edge')

    # Iterate over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Define the neighborhood
            i_min = i
            i_max = i + d
            j_min = j
            j_max = j + d

            # Extract the neighborhood
            W_p = padded_image[i_min:i_max, j_min:j_max]

            # Compute Gaussian spatial weights
            G = np.array([[gaussian(distance(i_min + x, j_min + y, i + pad_width, j + pad_width), sigmaSpace) 
                        for y in range(d)] for x in range(d)])

            # Compute the range weights
            F = gaussian(W_p - image[i, j], sigmaColor)

            # Compute the bilateral filter response
            W = G * F
            output[i, j] = np.sum(W * W_p) / np.sum(W)

    return output


def gaussian_kernel(size, sigma):
    """ Function to create a Gaussian kernel. """
    # Ensure that the size is odd and that we have an integer size for the kernel
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    # Normalize the kernel to ensure the sum is 1
    return g / g.sum()

def gaussian_filter(image, kernel_size, sigma):
    """ Function to apply Gaussian filter """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    
    # Create the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # The height and width of the kernel
    kh, kw = kernel.shape
    
    # Pad the image with reflection of the image (mirror effect)
    pad_height, pad_width = kh // 2, kw // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'reflect')
    
    filtered_image = convolve(padded_image, kernel, mode='reflect')
    
    return filtered_image




def laplacian(image):
    """ Function to apply Laplacian filter. """
    # Define the Laplacian kernel
    laplacian_kernel = np.array([[0, -1, 0],
                                [-1, 4, -1],
                                [0, -1, 0]], dtype=np.float64)

    # Pad the image to handle border pixels
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='reflect')

    # Initialize the output image
    laplacian_image = np.zeros_like(image, dtype=np.float64)

    # Apply the Laplacian kernel to each pixel in the image
    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            # Perform convolution
            region_of_interest = padded_image[i - 1:i + 2, j - 1:j + 2]
            laplacian_value = np.sum(region_of_interest * laplacian_kernel)
            laplacian_image[i - 1, j - 1] = laplacian_value

    # Convert to absolute values and then to 8-bit
    abs_laplacian = np.absolute(laplacian_image)
    abs_laplacian = np.uint8(abs_laplacian / np.max(abs_laplacian) * 255)

    return abs_laplacian


def sobel_filter(image):
    """ Function to apply Sobel filter. """
    # Define Sobel operator kernels.
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    # Apply Sobel operator kernels.
    sobelx = convolve(image, Kx)
    sobely = convolve(image, Ky)


    return sobelx, sobely

from skimage.filters import gaussian
# Function to perform edge detection
def edge_detection(image_path, kernel_size, threshold1, threshold2):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Smooth the image with Gaussian filter
    #smoothed_image = cv2.GaussianBlur(image, (kernel_size,kernel_size), 10)
    #smoothed_image = gaussian_filter(image, kernel_size=kernel_size, sigma=10)
    smoothed_image = gaussian(image, sigma=2, mode='reflect')
        
    sobelx, sobely = sobel_filter(smoothed_image)
    
    # grad = np.sqrt(sobelx**2 + sobely**2)
    # grad /= grad.max()
    # Calculate the gradient magnitude
    grad = np.sqrt(sobelx**2 + sobely**2)

    # Normalize to the range [0, 1]
    grad /= np.max(grad)
    
    # grad = sobel_filter(smoothed_image)
    
    edges0 = grad > threshold1 
    edges1 = grad > threshold2
    
    
    
    # Display images
    plt.figure(figsize=(15, 6))
    plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('Original Image') , plt.axis('off')
    plt.subplot(232), plt.imshow(smoothed_image, cmap='gray'), plt.title(f'Gaussian Smoothing {kernel_size}x{kernel_size}'), plt.axis('off')  
    plt.subplot(233), plt.imshow(edges0, cmap='gray'), plt.title(f'Edges threshold {threshold1}'), plt.axis('off')
    plt.subplot(234), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.axis('off')    
    plt.subplot(235), plt.imshow(smoothed_image, cmap='gray'), plt.title(f'Gaussian Smoothing {kernel_size}x{kernel_size}'), plt.axis('off')   
    plt.subplot(236), plt.imshow(edges1, cmap='gray'), plt.title(f'Edges threshold {threshold2}'), plt.axis('off')
    plt.show()


def hough_lines(edges, rho_resolution, theta_resolution, threshold):
    # Image dimensions
    height, width = edges.shape

    # Maximum possible rho value is the diagonal of the image
    max_rho = int(np.sqrt(height**2 + width**2))
    rhos = np.arange(-max_rho, max_rho, rho_resolution)
    thetas = np.arange(-np.pi, np.pi, theta_resolution)

    # Create the accumulator array to hold the values
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int64)

    # Indices of non-zero elements (edge points)
    edge_points = np.argwhere(edges != 0)

    # Hough Transform
    for y, x in edge_points:
        for theta_idx, theta in enumerate(thetas):
            # Calculate rho. "+ max_rho" is used for a positive index
            rho = int((x * np.cos(theta) + y * np.sin(theta)) + max_rho) // rho_resolution
            accumulator[rho, theta_idx] += 1

    # Thresholding and converting to (rho, theta)
    lines = []
    for rho_idx, theta_idx in np.argwhere(accumulator > threshold):
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        lines.append((rho, theta))

    return lines


# Function to apply Hough Transform and detect lines
def detect_lines(img, threshold, bin):
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    # lines = cv2.HoughLines(edges, 1, np.pi / bin, threshold)
    lines = hough_lines(edges, 1, np.pi / bin, threshold)
    
    img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        # for rho, theta in lines[:, 0]:
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_with_lines

# Create a synthetic image with lines and other shapes
def create_synthetic_image():
    # Start with an empty black image
    image = np.zeros((512, 512, 3), dtype=np.uint8)

    # Draw white lines
    cv2.line(image, (100, 100), (400, 100), (255, 255, 255), thickness=5)
    cv2.line(image, (100, 200), (400, 150), (255, 255, 255), thickness=5)
    # Draw a white circle
    cv2.circle(image, (256, 256), 50, (255, 255, 255), thickness=-1)
    # Draw a white rectangle
    cv2.rectangle(image, (150, 300), (350, 400), (255, 255, 255), thickness=-1)

    return image


# Function to add noise to the image
def add_noise(image, amount=500):

    noisy_image = image.copy()

    # Randomly choose pixels to add noise
    for _ in range(amount):
        x_coord = np.random.randint(0, image.shape[1])
        y_coord = np.random.randint(0, image.shape[0])
        noisy_image[y_coord, x_coord] = 255  # Set the pixel to white to simulate noise

    return noisy_image

# Function to add occlusion to the image
def add_occlusion(image, top_left, bottom_right):

    occluded_image = image.copy()

    # Draw a black rectangle over the image to simulate occlusion
    rr, cc = draw.rectangle(start=top_left, end=bottom_right)
    occluded_image[rr, cc] = 0  # Set the pixels to black to simulate occlusion

    return occluded_image


def plot_image_with_line(images, thresholds, bins):
    for bin in bins:
        plt.figure(figsize=(15, 6))
        for i, threshold in enumerate(thresholds, 1):
            img_with_lines = detect_lines(images, threshold, bin)
            plt.subplot(1, 3, i)
            plt.imshow(img_with_lines)
            plt.title(f'Hough Lines with threshold: {threshold} bin: {bin}')
            plt.axis('off')
        plt.show()

def main():

##### Q1: #####
    # Load the noisy image
    print("Question number 1")
    
    image_path = './Images/building_noisy.png'
    building_noisy_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply bilateral and Gaussian filters
    bilateral_filtered_img = bilateral_filter(building_noisy_img, d=3, sigmaColor=75, sigmaSpace=75)
    gaussian_smoothed_img = gaussian_filter(building_noisy_img, kernel_size=7, sigma=10)


    # Display results
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1), plt.imshow(building_noisy_img, cmap='gray'), plt.title(' Noisy Image'), plt.axis('off')
    plt.subplot(1, 3, 2), plt.imshow(bilateral_filtered_img, cmap='gray'), plt.title('Bilateral Filtered Image'), plt.axis('off')
    plt.subplot(1, 3, 3), plt.imshow(gaussian_smoothed_img, cmap='gray'), plt.title(' Gaussian Smoothed Image'), plt.axis('off')
    plt.show()
    
    # Apply Laplacian filter
    laplacian_original = laplacian(building_noisy_img)
    laplacian_bilateral = laplacian(bilateral_filtered_img)
    laplacian_gaussian = laplacian(gaussian_smoothed_img)

    # Display results
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1), plt.imshow(laplacian_original, cmap='gray'), plt.title('Laplacian of Noisy Image'), plt.axis('off')
    plt.subplot(1, 3, 2), plt.imshow(laplacian_bilateral, cmap='gray'), plt.title('Laplacian of Bilateral Filtered Image'), plt.axis('off')
    plt.subplot(1, 3, 3), plt.imshow(laplacian_gaussian, cmap='gray'), plt.title('Laplacian of Gaussian Smoothed Image'), plt.axis('off')
    plt.show()
    
##### Q2: #####
    # Define parameters
    print("Question 2 start")
    gaussian_kernel_size = 3
    
    threshold1 = 0.1
    threshold2 = 0.3

    # Apply edge detection to each image
    for image_name in ['./Images/book_noisy1.png', './Images/book_noisy2.png', './Images/architecture_noisy1.png', './Images/architecture_noisy2.png']:
        edge_detection(image_name, gaussian_kernel_size, threshold1, threshold2)
        
        
        
    # change gaussion filter size 
    gaussian_kernel_size = 7
    # Apply edge detection to each image
    for image_name in ['./Images/book_noisy1.png', './Images/book_noisy2.png', './Images/architecture_noisy1.png', './Images/architecture_noisy2.png']:
        edge_detection(image_name, gaussian_kernel_size,  threshold1, threshold2)

        
    
        
        
##### Q3: #####
    print("Question 3 start")
    # Varying thresholds for Hough Transform
    thresholds = [50, 100, 150]

    # Varying bins for Hough Transform
    bins = [45, 90, 180]

    synthetic_image = create_synthetic_image()
    si = cv2.cvtColor(synthetic_image, cv2.COLOR_BGR2GRAY)
    
    # Add noise and occlusion to the synthetic image
    noisy_image = add_noise(si, amount=2500)
    occluded_image = add_occlusion(si, top_left=(50, 200), bottom_right=(350, 270))
    

    plt.figure(figsize=(15, 6))
    plt.subplot(131), plt.imshow(si, cmap='gray'), plt.title('syntheic image')    
    plt.subplot(132), plt.imshow(noisy_image, cmap='gray'), plt.title('added noise')    
    plt.subplot(133), plt.imshow(occluded_image, cmap='gray'), plt.title('added Occlusion')
    plt.show()

    # Display the results with different thresholds
    image = './Images/building_noisy.png'
    real_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    real_image= cv2.GaussianBlur(real_img, (5,5), 2)
    plot_image_with_line(si, thresholds, bins)
    plot_image_with_line(noisy_image, thresholds, bins)
    plot_image_with_line(occluded_image, thresholds, bins)
    plot_image_with_line(real_image, thresholds, bins)
    
    print("Assignment done!!!")
        
if __name__ == '__main__':
    main()