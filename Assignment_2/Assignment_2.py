import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from skimage.io import imread
from scipy.ndimage import convolve


###### Q2 ########
def gamma_transform(image, gamma):
    img = (image)/255
    g_img = (img)**gamma
    return g_img*255

def histogram_equalize(image):
    h = image.ravel()
    avg_intensity = h.mean()
    print(f"average intensity: {avg_intensity:.3f}")
    freq = np.zeros(256, dtype=int)
    MN = len(h)
    for i in h:
        freq[i]+=1
    plt.hist(h,bins=256,range=[0,256])
    plt.title('histogram before equalization' )
    plt.show()
    prob = freq/MN 
    F = []
    for i in range(0,256):
        F.append(prob[:i].sum())
        
    mn = min(F)
    mx = max(F)
    F = (F - mn)/(mx - mn)
    e_image = image/255
    for i in range(e_image.shape[0]):
        for j in range(e_image.shape[1]):
            e_image[i][j]=F[image[i][j]]
            
    h2 = e_image.ravel()
    plt.hist(h2,256,[0,1])
    plt.title('histogram equalized')
    plt.show()
    return e_image*255

def mse(img1, img2):
    #mean sqaured error
    N = img1.size
    error = (img1-img2)**2
    m_error = np.sum(error)/N
    return m_error

###### Q3 ######## 
def nearest_neighbor(image, x_rot, y_rot):
    #dimensions
    height = image.shape[0]
    width = image.shape[1]
    n_height = x_rot.shape[0]
    n_width = x_rot.shape[1]
    
    #nearest neighbor by taking closed integer
    n_x = x_rot.astype(int)
    n_y = y_rot.astype(int)

    rot_img = np.ones((n_height, n_width)) * 255

    for i in range(n_height):
        for j in range(n_width):
            
            if((n_x[i,j] >= 0) and (n_x[i,j] < width) and (n_y[i,j] >= 0) and (n_y[i,j] < height)):
                
                rot_img[i,j] = image[n_y[i,j],n_x[i,j]]
                
    return rot_img  

def bilinear(image, x_rot, y_rot):
    
    height = image.shape[0]
    width = image.shape[1]
    
    n_height = x_rot.shape[0]
    n_width = x_rot.shape[1]
    x1 = np.floor(x_rot).astype(int)
    x2 = np.ceil(x_rot).astype(int)
    y1 = np.floor(y_rot).astype(int)
    y2 = np.ceil(y_rot).astype(int)

    rot_img = np.ones((n_height, n_width)) * 255
    wx = x_rot - x1
    wy = y_rot - y1

    for i in range(n_height):
        for j in range(n_width):
            
            if((x1[i,j] >= 0) and (x2[i,j] < width) and (y1[i,j] >= 0) and (y2[i,j] < height)):
                
                wxr = wx[i,j].reshape(-1,1)
                wyr = wy[i,j].reshape(-1,1)
                
                a1 = image[y1[i,j],x1[i,j]]
                a2 = image[y1[i,j],x2[i,j]]
                a3 = image[y2[i,j],x1[i,j]]
                a4 = image[y2[i,j],x2[i,j]]
                
                rot_img[i,j] = (a1*(1 - wxr)+ a2*wxr)*(1 - wyr) + (a3*(1 - wxr)+ a4*wxr)*wyr
                
    return rot_img  

def rotate_image(image, degree, method):
    

    t=degree*(np.pi/180)
    #dimensions
    height = image.shape[0]
    width = image.shape[1]
    centre_h = height/2
    centre_w = width/2

    corners = [(0,0),(height-1,0),(0,width-1),(height-1,width-1)]
    new_corner = []
    for x in corners:
        a1  = (x[0])*np.cos(t) - (x[1])*np.sin(t) 
        a2 = (x[0])*np.sin(t) + (x[1])*np.cos(t) 
        new_corner.append([a1,a2])
        
    n = np.array(new_corner)
    n_height = int(max(n[:,0])-min(n[:,0]))+1 ##rows
    n_width = int(max(n[:,1])-min(n[:,1]))+1 ##columns

    #build grid of coordinates
    y_grid,  x_grid = np.indices((n_height,n_width))

    #shift origin
    y_grid_c , x_grid_c = y_grid - n_height/2, x_grid - n_width/2

    #rotate around origin and shift origin
    x_rot = (x_grid_c)*np.cos(t) - (y_grid_c)*np.sin(t) + centre_w
    y_rot = (x_grid_c)*np.sin(t) + (y_grid_c)*np.cos(t) + centre_h
    
    if(method=='NN'):
        return nearest_neighbor(image,x_rot,y_rot)
    else:
        return bilinear(image,x_rot,y_rot)


###### Q4 ######## 
def high_boost_filter(image, scale):
    #filter
    kernel1 = np.ones((3, 3)) / 9
    kernel2 = np.ones((5, 5)) / 25

    #blur image
    bimg1 = cv.filter2D(src=image, ddepth=-1, kernel=kernel1)
    bimg2 = cv.filter2D(src=image, ddepth=-1, kernel=kernel2)
    #mask
    mask1 = image - bimg1
    mask2 = image - bimg2
    
    img1 = image + scale* mask1
    img2 = image + scale* mask2
            
    return img1, img2

    


def main():
    
    print("Code started execution")
    #Question 1
    ECE = imread('Images/ECE.png')
    plt.figure(figsize=(10,6))

    plt.subplot(1, 2, 1)
    plt.imshow(ECE,cmap='gray')
    plt.title("Default Image")

    plt.subplot(1, 2, 2)
    plt.title("Correct Image")
    plt.imshow(ECE,cmap='gray',vmin=0, vmax=255)
    plt.show()
    
    #Question 2
    hazy= cv.imread('Images/hazy.png',0)
    e_img = histogram_equalize(hazy)
    
    plt.imshow(e_img,cmap='gray')
    plt.title(f'Histogram Equalized Image')
    plt.show()
    
    min_mse = float('inf')
    best_gamma = 0.0
    step_size = 0.01
    for g in np.arange(0.1, 5, step_size):
        g_img = gamma_transform(hazy,g)
        curr_mse = mse(e_img,g_img)
        
        if curr_mse < min_mse:
            min_mse = curr_mse
            best_gamma = g
            
    best_g_img = gamma_transform(hazy, best_gamma) 
    h = best_g_img.ravel()
    plt.hist(h,256,[0,255])
    plt.title('histogram of gamma corrected image ')
    plt.show()
    
    plt.figure(figsize=(13,5))
    plt.subplot(1, 3, 1)
    plt.imshow(hazy,cmap='gray',vmin=0, vmax=255)
    plt.title("original image",)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("equalized image")
    plt.imshow(e_img,cmap='gray',vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"gamma corrected(g = {best_gamma:.3f})")
    plt.imshow(best_g_img,cmap='gray',vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    #Question 3
    box = cv.imread('Images/box.png',0) 
    degree = [-5,30]
    for d in degree:
        
        r_img_NN = rotate_image(box,d,'NN')
        r_img_BI= rotate_image(box,d,'BI')
        
        plt.figure(figsize=(10,6))
        plt.subplot(1, 2, 1)
        plt.imshow(r_img_NN,cmap='gray')
        plt.title(f"nearest neighbor, rotation({d}°)",)

        plt.subplot(1, 2, 2)
        plt.imshow(r_img_BI,cmap='gray')
        plt.title(f"bilinear interpolation, rotation({d}°)")
        plt.show() 
        
    #Question 4   
    study = cv.imread('Images/study.png',0)
    k = 2.5
    img1, img2 = high_boost_filter(study, k)

    plt.figure(figsize=(10,6))
    plt.subplot(1, 2, 1)
    plt.imshow(img1,cmap='gray')
    plt.title(f"High boost filter(3x3)",)

    plt.subplot(1, 2, 2)
    plt.imshow(img2,cmap='gray')
    plt.title(f"high boost filter(5X5)")
    plt.show()
    print("Execution completed!!")
    

if __name__ == '__main__':
    main()