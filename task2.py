
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

def findIntrinsicParams(world_cords,img_points):
    A1=[]
    A2=[]
    for item in world_cords:
        litem=list(item)
        l2=list(item)
        litem2=[]
        litem.append(1)
        litem.append(0)
        litem.append(0)
        litem.append(0)
        litem.append(0)
        A1.append(litem)
        litem2.append(0)
        litem2.append(0)
        litem2.append(0)
        litem2.append(0)
        litem2.extend(l2)
        litem2.append(1)
        A2.append(litem2)
    A=[x for y in zip(A1,A2) for x in y]
    A=np.array([np.array(x) for x in A],dtype=object)   
    A5=[]
    img_points=np.array(img_points).flatten().reshape(36,2)
    for i in range(0,36):
        x=img_points[i]
        w=world_cords[i]
        A3=np.array([-x[0]*w[0], -x[0]*w[1],-x[0]*w[2],-x[0]])
        A4=np.array([-x[1]*w[0], -x[1]*w[1],-x[1]*w[2],-x[1]])
        A5.append(A3)
        A5.append(A4)
    A5=np.array(A5).flatten().reshape(72,4)
    A6=np.hstack((A,A5)).astype(float)
    # print(A6)
    U,S,VT=np.linalg.svd(A6)
    m=VT[11].reshape((3,4))
    m=m/(np.linalg.norm(np.array([m[2][0],m[2][1],m[2][2]])))
    m1=np.array([m[0][0],m[0][1],m[0][2]]).T
    m2=np.array([m[1][0],m[1][1],m[1][2]]).T
    m3=np.array([m[2][0],m[2][1],m[2][2]]).T
    m4=np.array([m[0][3],m[1][3],m[2][3]]).T
    ox=np.dot(m1.T,m3)
    oy=np.dot(m2.T,m3)
    fx=np.sqrt((np.dot(m1.T,m1)-ox**2))
    fy=np.sqrt((np.dot(m2,m2.T)-oy**2))

    # cc=np.dot(m,np.array([40,0,40,1]))
    # print(cc[0]/cc[2],cc[1]/cc[2])

    return [fx,fy,ox,oy]

def calibrate(imgname):
    #......
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)    
    load_image=imread(imgname)
    gray_image=cvtColor(load_image,COLOR_BGR2GRAY)    
    r,corners = findChessboardCorners(gray_image,(4,9))
    img_points = []    
    corners2 = cornerSubPix(gray_image,corners, (6,6), (-1,-1), criteria)
    img_points.append(corners)
    drawChessboardCorners(load_image, (4,9), corners2, r)
    
    # print(img_points[0])   
    world_cords=np.array([
        [40,0,40],[40,0,30],[40,0,20],[40,0,10],
        [30,0,40],[30,0,30],[30,0,20],[30,0,10],
        [20,0,40],[20,0,30],[20,0,20],[20,0,10],
        [10,0,40],[10,0,30],[10,0,20],[10,0,10],
        [0,0,40],[0,0,30],[0,0,20],[0,0,10],
        [0,10,40],[0,10,30],[0,10,20],[0,10,10],
        [0,20,40],[0,20,30],[0,20,20],[0,20,10],
        [0,30,40],[0,30,30],[0,30,20],[0,30,10],
        [0,40,40],[0,40,30],[0,40,20],[0,40,10]
    ]) 
    
    intrinsic_params=findIntrinsicParams(world_cords,img_points)
    # print(intrinsic_params)
    return intrinsic_params,True   

if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)
