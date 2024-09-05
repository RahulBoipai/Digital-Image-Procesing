import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time


#####------ Q1 -------#####
def Histogram(image):
    h = image.ravel()
    avg_intensity = h.mean()
    print(f"average intensity: {avg_intensity:.3f}")
    freq = np.zeros(256, dtype=int)

    for i in h:
        freq[i]+=1
    plt.hist(h,256,[0,256])
    plt.title('Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.show()
    return freq

#####------ Q2 -------#####
def means(i,j ,p, w):
    m=0
    for k in range(i,j):
        m += k*p[k]
    return m/w

def variances(i,j,p,w,m):
    v=0
    for k in range(i,j):
        v+=(k-m)**2*p[k]
    return v/w

def within_class_variance(image,t):
    h = image.ravel()
    MN = len(h)
    freq = np.zeros(256, dtype=int)
    for i in h:
        freq[i]+=1
        
    p = freq/MN
    w0 = p[:t+1].sum()
    if(np.isclose(w0,0.0)):
        return float('inf')
    
    m0 = means(0,t+1,p,w0)
    v0 = variances(0,t+1,p,w0,m0)

    c1 = freq[t+1:]
    w1 = p[t+1:].sum()
    if(np.isclose(w1,0.0)):
        return float('inf')
        
    m1 = means(t+1,256,p,w1)
    v1 = variances(t+1,256,p,w1,m1)
    
    return w0*v0 + w1*v1

def between_class_variance(image,t):
    h = image.ravel()
    MN = len(h)
    freq = np.zeros(256, dtype=int)
    for i in h:
        freq[i]+=1
        
    p = freq/MN
    w0 = p[:t+1].sum()
    mT = means(0,256,p,1)
    vT = variances(0,256,p,1,mT)
    
    if(np.isclose(w0,0.0)):
        return 0.0
    
    m0 = means(0,t+1,p,w0)
    v0 = variances(0,t+1,p,w0,m0)

    c1 = freq[t+1:]
    w1 = p[t+1:].sum()
    if(np.isclose(w1,0.0)):
        return 0.0
        
    m1 = means(t+1,256,p,w1)
    v1 = variances(t+1,256,p,w1,m1)
    
    return w0*(m0-mT)**2 + w1*(m1-mT)**2

def Otsus_Binarization(image):
    Vw = np.zeros(256)
    Vb= np.zeros(256)
    start = time.time()
    for i in range(0,256):
        Vw[i]=within_class_variance(image,i)
    t1 = Vw.argmin()
    end = time.time()
    print(f"Time taken for within class: {end - start:.3f}")
    print(f"Optimal Threshold by minimizing within class variance: {t1} ")
    
    start = time.time()
    for i in range(0,256):
        Vb[i]=between_class_variance(image,i)
    t2 = Vb.argmax()   
    end = time.time()
    print(f"Time taken for between class: {end - start:.3f}")
    print(f"Optimal Threshold by maximizing between class variance: {t2} ")
    return t1
    


#####------ Q3 -------#####
def Image_superimpose(text, depth, background):
    imgx=cv.cvtColor(text,cv.COLOR_BGR2RGB)
    imgy=cv.cvtColor(depth,cv.COLOR_BGR2RGB)
    imgz=cv.cvtColor(background,cv.COLOR_BGR2RGB)
    
    
    th = Otsus_Binarization(imgy) #around 150
    _, hst = cv.threshold(imgy, th, 255, cv.THRESH_BINARY)
    
    dst=hst/255
    conv0 = imgx.copy()
    conv1 = imgz.copy()
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            for k in range(dst.shape[2]):
                if(dst[i][j][k]==1): 
                    conv0[i][j][k]=imgx[i][j][k]
                    conv1[i][j][k]=imgx[i][j][k]
                
                else:
                    conv0[i][j][k]=0
    plt.imshow(imgx)
    plt.show()
    plt.imshow(conv0)
    plt.show()
    plt.imshow(imgz)
    plt.show()
    plt.imshow(conv1)
    plt.show()
    return conv1

#####------ Q4 -------#####
def count_character(image):
    th = Otsus_Binarization(image)
    _, grid = cv.threshold(image, th, 255, cv.THRESH_BINARY)
    from collections import deque
    rows, cols = grid.shape[0], grid.shape[1]
    visited=set()
    islands=0

    def bfs(r,c,s):
        q = deque()
        visited.add((r,c))
        q.append((r,c))

        while q:
            row,col = q.popleft()
            directions= [[1,0],[-1,0],[0,1],[0,-1]]
        
            for dr,dc in directions:
                r,c = row + dr, col + dc
                if (r) in range(rows) and (c) in range(cols) and grid[r][c] == 0 and (r ,c) not in visited:
                    q.append((r , c ))
                    s+=1
                    visited.add((r, c ))
        return s

    size = [ ]
    for r in range(rows):
        for c in range(cols):
        
            if grid[r][c] == 0 and (r,c) not in visited:
                
                s=bfs(r,c,0)
                size.append(s)
    count=0
    size = np.array(size)
    mean_size = size.mean()
    for i in size:
        if i > mean_size/3:
            count +=1 
    return count

#####------ Q5 -------#####

def MSER(image):
    
    Char_count = []
    for i in range(1,256):
        _, grid = cv.threshold(image, i, 255, cv.THRESH_BINARY)
        h = grid.ravel()
        freq = np.zeros(256, dtype=int)
        for i in h:
            freq[i]+=1
        intensity = 255-freq.argmax()
        
        from collections import deque
        rows, cols = grid.shape[0], grid.shape[1]
        visited=set()
        count=0

        def bfs(r,c,s):
            q = deque()
            visited.add((r,c))
            q.append((r,c))

            while q:
                row,col = q.popleft()
                directions= [[1,0],[-1,0],[0,1],[0,-1]]
            
                for dr,dc in directions:
                    r,c = row + dr, col + dc
                    if (r) in range(rows) and (c) in range(cols) and grid[r][c] == intensity and (r ,c) not in visited:
                        q.append((r , c ))
                        s+=1
                        visited.add((r, c ))
            return s

        for r in range(rows):
            for c in range(cols):
            
                if grid[r][c] == intensity and (r,c) not in visited:
                    s=bfs(r,c,0)
                    if s>10000: 
                        count +=1
        Char_count.append(count)
    return Char_count

def main():
    
    #question1
    print("  Q1, Histogram Computation")
    imgC = cv.imread('./Images/coins.png',0)
    plt.imshow(imgC,'gray')
    plt.show()
    H = Histogram(imgC)
    print("frequency at each intensity level:")
    print(H)
    
    #question2
    print("  Q2, Otsu's Binarization")
    t = Otsus_Binarization(imgC)
    _, dst = cv.threshold(imgC, t, 255, cv.THRESH_BINARY)
    plt.imshow(imgC,'gray')
    plt.show()
    plt.imshow(dst,'gray')
    plt.show()
    
    #question3
    print("  Q3, Depth based extraction")
    text = cv.imread('./Images/IIScText.png')
    depth = cv.imread('./Images/IIScTextDepth.png')
    background = cv.imread('./Images/IIScMainBuilding.png')
    img_s=Image_superimpose(text, depth, background)
    
    #question4
    print("  Q4, Connected Components")
    imgQ = cv.imread('./Images/quote.png',0)
    
    count=count_character(imgQ)
    plt.imshow(imgQ,'gray')
    plt.show()
    print("No. of character found: ",count)
    
    #question5
    print("  Q5, MSER")
    img_char = cv.imread('./Images/Characters.png',0)
    
    char_count=MSER(img_char)
    plt.imshow(img_char,'gray')
    plt.show()
    print(char_count)
     
    
if __name__ == '__main__':
    main()
    