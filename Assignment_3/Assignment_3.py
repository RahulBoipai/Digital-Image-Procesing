
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import cv2 as cv


###### Q1 ########
def radial_sinusoid_image(M,f0):
    '''M : side length , f0: frequency '''
    #create 2d grid of values
    x = np.linspace(-M/2, M/2, M)
    y = np.linspace(-M/2, M/2, M)
    u,v = np.meshgrid(x,y)
    #distance from centre
    D = np.sqrt(u**2 + v**2)
    #sinusoid
    image = np.cos(2*np.pi *f0*D/M)
    return image

def DFT(image):
    #compute DFT using FFT
    dft_image = np.fft.fft2(image)
    #cyclically shift to centre
    shift_dft_image = np.fft.fftshift(dft_image)
    #magnitude of dft
    magnitude = np.abs(shift_dft_image)
    log_dft = np.log1p(magnitude)
    return log_dft, dft_image

def IDFT(dft_image):
    recon_img = np.fft.ifft2(dft_image).real
    return recon_img


###### Q2 ########
def ILPF(image, D0):
    p , q = image.shape
    F_image = np.fft.fft2(image)
    F_shift = np.fft.fftshift(F_image)
    u , v = np.arange(p),np.arange(q)

    #filter H
    D = np.sqrt((u-p/2)**2 + (v-q/2)**2)
    H = []
    for d in D:
        if(d<=100):
            H.append(1)
        else:
            H.append(0)
            
    filtered = F_shift*H
    filter_img = np.fft.ifft2(np.fft.ifftshift(filtered)).real
    return filter_img

def GLPF(image,D0):
    p , q = image.shape
    F_image = np.fft.fft2(image)
    F_shift = np.fft.fftshift(F_image)
    u , v = np.arange(p),np.arange(q)
    #filter H
    D = np.sqrt((u-p/2)**2 + (v-q/2)**2)
    H = np.exp(-(D**2)/(2*(D0**2)))
    
    filtered = F_shift*H
    filter_img = np.fft.ifft2(np.fft.ifftshift(filtered)).real
    return filter_img
    

###### Q3 ######## 
def inverse_filter(blur_image, kernel):
    m,n = kernel.shape
    #shift and FTon kernel
    s_kernel = np.roll(kernel, (int(m/2), int(n/2)), axis=(0,1))
    F_kernel = np.fft.fft2(s_kernel, s=blur_image.shape)
    
    #frequecy domain image
    F_blur = np.fft.fft2(blur_image)
    F_deblur = np.where(np.abs(F_kernel)>0.1, F_blur/F_kernel, 0)
    
    #inverse FT
    deblur_image = np.fft.ifft2(F_deblur).real
    return deblur_image
    
def wiener_filter(blur_image,kernel,sd):
    k=1e5
    M,N = blur_image.shape
    #FT on image
    F_blur = np.fft.fft2(blur_image)
    m,n = kernel.shape
    #shift and FT on kernel 
    s_kernel = np.roll(kernel, (int(m/2), int(n/2)), axis=(0,1))
    F_kernel = np.fft.fft2(s_kernel, s=blur_image.shape)
    u = np.fft.fftfreq(M)
    v = np.fft.fftfreq(N)

    U,V = np.meshgrid(u,v)
    sf = k/ (U**2 + V**2+1) #add 1 to remove divide by zero error
    sn = sd**2
    W_filter = np.conj(F_kernel)/(np.abs(F_kernel)**2 + sn/sf)
    
    deblur_image = np.fft.ifft2(W_filter*F_blur).real
    return deblur_image

    

def main():
    '''Question 1'''
    M = 500  #size of image MxM
    frequency = [2,10,20,50,100] #multiple frequecies
    
    for f0 in frequency:
        image = radial_sinusoid_image(M,f0)
        img, dft_img = DFT(image)
        idft_img = IDFT(dft_img)

        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"radial sinusoidal image, f0:{f0}")

        plt.subplot(1, 3, 2)
        plt.imshow(img, cmap='gray')
        plt.title(f"DFT of image, f0:{f0}")
        
        plt.subplot(1, 3, 3)
        plt.title("Reconstructed Image")
        plt.imshow(idft_img,cmap='gray')
        plt.show()
    
    '''Question 2'''
    character = cv.imread('./Files/characters.tif',0)
    D0=100
    ilpf_img = ILPF(character, D0)
    glpf_img = GLPF(character, D0) 
    
    plt.figure(figsize=(6,5))

    plt.subplot(1,1,1)
    plt.imshow(character, cmap='gray',vmin=0,vmax=255)
    plt.title(f"Original image")
    plt.show()

    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(ilpf_img, cmap='gray',vmin=0, vmax=255)
    plt.title(f"ILPF image")
    
    plt.subplot(1, 2, 2)
    plt.title("GLPF image")
    plt.imshow(glpf_img,cmap='gray',vmin=0,vmax=255)
    plt.show()
     
    '''Question 3'''
    blr_low = cv.imread('./Files/Blurred_LowNoise.png',0)
    blr_high = cv.imread('./Files/Blurred_HighNoise.png',0)
    data = loadmat('./Files/BlurKernel.mat')
    kernel = data['h']
    
    #Low Blur image
    if_unblur_low = inverse_filter(blr_low, kernel)
    wf_unblur_low = wiener_filter(blr_low,kernel,1)
    
    #High Blur image
    if_unblur_high = inverse_filter(blr_high, kernel)
    wf_unblur_high = wiener_filter(blr_high,kernel,10)
    
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(blr_low, cmap='gray',vmin=0,vmax=255)
    plt.title(f"Low Blur Image")
    plt.subplot(1, 2, 2)
    plt.imshow(if_unblur_low, cmap='gray',vmin=0,vmax=255)
    plt.title(f"Unblur Image (inverse filter)")
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(blr_low, cmap='gray',vmin=0,vmax=255)
    plt.title(f"Low Blur Image")
    plt.subplot(1, 2, 2)
    plt.imshow(wf_unblur_low, cmap='gray',vmin=0,vmax=255)
    plt.title(f"Unblur Image (wiener filter)")
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(blr_high, cmap='gray',vmin=0,vmax=255)
    plt.title(f"High Blur Image")
    plt.subplot(1, 2, 2)
    plt.imshow(if_unblur_high, cmap='gray',vmin=0,vmax=255)
    plt.title(f"Unblur Image (inverse filter)")
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(blr_high, cmap='gray',vmin=0,vmax=255)
    plt.title(f"High Blur Image ")
    plt.subplot(1, 2, 2)
    plt.imshow(wf_unblur_high, cmap='gray',vmin=0,vmax=255)
    plt.title(f"Unblur Image (wiener filter)")
    plt.show()

if __name__ == '__main__':
    main()