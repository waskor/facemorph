import dlib 
import cv2
import numpy as np
import sys

def find_landmarks(images, lm_list):
    global height, width
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    lm_no = 68

    height, width, channels = images[0].shape

    for p in range(len(images)):

        #intialise coordinate array for facial landmarks + corners + edges
        lm_array = np.empty((lm_no+8,2))
    
        #detect border of face, upsamples image 1 time
        rects = detector(images[p], 1)

        #find facial landmarks
        shape = predictor(images[p], rects[0]) 

        #insert found landmarks into array
        for i in range(lm_no):
            x = shape.part(i).x
            y = shape.part(i).y
            lm_array[i][0] = x 
            lm_array[i][1] = y
            
            #draw circle on found landmark:
            #cv2.circle(images[p], (x,y), 3, (255, 0, 0), -1)
            
        #insert coordinates for corners + edges
        lm_array[lm_no] = (1,1)
        lm_array[lm_no+1] = (width-1,1)
        lm_array[lm_no+2] = (width/2,1)
        lm_array[lm_no+3] = (1,height/2)
        lm_array[lm_no+4] = (1,height-1)
        lm_array[lm_no+5] = (width-1, height-1)
        lm_array[lm_no+6] = (width-1, height/2)
        lm_array[lm_no+7] = (width/2, height-1)
        
        #store landmarks for image in list
        lm_list.append(lm_array)

def landmark_triangulation(images, lm_list, triangle_list):

    #Delaunay triangulation for image 1
            
    #initialise empty Delaunay subdivision
    delrect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(delrect)

    #insert points into the created subdivision
    for i in range(lm_list[0].shape[0]):
        subdiv.insert((lm_list[0][i][0], lm_list[0][i][1]))

    #apply delaunay triangulation to subdivision
    triangles = subdiv.getTriangleList()

    #store image 1 triangles in list
    triangle_list.append(triangles)

    #draw triangles
    for t in range(triangles.shape[0]):

        t1 = (triangles[t,0], triangles[t,1])
        t2 = (triangles[t,2], triangles[t,3])
        t3 = (triangles[t,4], triangles[t,5])

        # cv2.line(images[p], t1, t2, (255, 0, 0), 1)
        # cv2.line(images[p], t2, t3, (255, 0, 0), 1)
        # cv2.line(images[p], t3, t1, (255, 0, 0), 1)

    #Triangulation for image 2 based on image 1 triangulation

    #initialise array for triangles in image 2
    trianglesnew = np.zeros((triangle_list[0].shape[0], 6))
    
    #loop through each triangle found in image 1
    for t in range(triangle_list[0].shape[0]):
        
        t1 = (triangles[t,0], triangles[t,1])
        t2 = (triangles[t,2], triangles[t,3])
        t3 = (triangles[t,4], triangles[t,5])
        
        #determine index of triangle vertices
        index1 = np.where((lm_list[0][:,0] == t1[0]) & (lm_list[0][:,1] == t1[1]))[0][0]
        index2 = np.where((lm_list[0][:,0] == t2[0]) & (lm_list[0][:,1] == t2[1]))[0][0]
        index3 = np.where((lm_list[0][:,0] == t3[0]) & (lm_list[0][:,1] == t3[1]))[0][0]

        #create triangle in image 2 based on the deteremined indices
        trianglenew = np.array([[lm_list[1][index1,0], lm_list[1][index1,1], lm_list[1][index2,0], lm_list[1][index2,1], lm_list[1][index3,0], lm_list[1][index3,1]]]).astype(np.float32)

        #set new vertices
        t1 = (trianglenew[0,0], trianglenew[0,1])
        t2 = (trianglenew[0,2], trianglenew[0,3])
        t3 = (trianglenew[0,4], trianglenew[0,5])

        #draw triangles
        # cv2.line(images[p], t1, t2, (255, 0, 0), 1)
        # cv2.line(images[p], t2, t3, (255, 0, 0), 1)
        # cv2.line(images[p], t3, t1, (255, 0, 0), 1)

        #insert triangle into image 2 array
        trianglesnew[t,:] = trianglenew[0,:]

        #store image 2 triangles in list
        triangle_list.append(trianglesnew)

def morph_triangles(triangle_list, img1, img2, img3, k):
    no_triangles = triangle_list[0].shape[0]
    #initialise output image array


    for n in range(no_triangles):
        #find intermediate triangle by interpolating between triangle in image 1 and image 2
        triangle_intr = np.array([
            [triangle_list[0][n,0]+(triangle_list[1][n,0]-triangle_list[0][n,0])*k , triangle_list[0][n,1]+(triangle_list[1][n,1]-triangle_list[0][n,1])*k],
            [triangle_list[0][n,2]+(triangle_list[1][n,2]-triangle_list[0][n,2])*k , triangle_list[0][n,3]+(triangle_list[1][n,3]-triangle_list[0][n,3])*k],
            [triangle_list[0][n,4]+(triangle_list[1][n,4]-triangle_list[0][n,4])*k , triangle_list[0][n,5]+(triangle_list[1][n,5]-triangle_list[0][n,5])*k]
            ]).astype(np.float32)

        #set triangles
        triangle_2 = np.array([[triangle_list[1][n,0],triangle_list[1][n,1]],[triangle_list[1][n,2],triangle_list[1][n,3]],[triangle_list[1][n,4],triangle_list[1][n,5]]]).astype(np.float32)
        triangle_1 = np.array([[triangle_list[0][n,0],triangle_list[0][n,1]],[triangle_list[0][n,2],triangle_list[0][n,3]],[triangle_list[0][n,4],triangle_list[0][n,5]]]).astype(np.float32)
        
        #find the bounding rectangle of each triangle
        tri_r1 = cv2.boundingRect(triangle_1)
        tri_ri = cv2.boundingRect(triangle_intr)
        tri_r2 = cv2.boundingRect(triangle_2)
        
        #crop image to bounding rectangle
        imgrect1 = img1[tri_r1[1]:tri_r1[1]+tri_r1[3], tri_r1[0]:tri_r1[0]+tri_r1[2]]
        imgrect2 = img2[tri_r2[1]:tri_r2[1]+tri_r2[3], tri_r2[0]:tri_r2[0]+tri_r2[2]]
        
        #set new coordinates of triangles within the cropped region
        triangle_1_crop = np.array([[triangle_1[0][0]-tri_r1[0], triangle_1[0][1] - tri_r1[1]],[triangle_1[1][0]-tri_r1[0], triangle_1[1][1] - tri_r1[1]],[triangle_1[2][0]-tri_r1[0], triangle_1[2][1] - tri_r1[1]]]).astype(np.float32)
        triangle_2_crop = np.array([[triangle_2[0][0]-tri_r2[0], triangle_2[0][1] - tri_r2[1]],[triangle_2[1][0]-tri_r2[0], triangle_2[1][1] - tri_r2[1]],[triangle_2[2][0]-tri_r2[0], triangle_2[2][1] - tri_r2[1]]]).astype(np.float32)
        triangle_intr_crop = np.array([[triangle_intr[0][0]-tri_ri[0], triangle_intr[0][1] - tri_ri[1]],[triangle_intr[1][0]-tri_ri[0], triangle_intr[1][1] - tri_ri[1]],[triangle_intr[2][0]-tri_ri[0], triangle_intr[2][1] - tri_ri[1]]]).astype(np.float32)

        #find the affine transformation matrix
        affine1 = cv2.getAffineTransform(triangle_1_crop, triangle_intr_crop)
        affine2 = cv2.getAffineTransform(triangle_2_crop, triangle_intr_crop)

        #initialise mask with size of intermediate triangle
        mask = np.zeros((tri_ri[3], tri_ri[2])).astype(np.uint8)
        #fill mask with intermediate triangle
        tri = cv2.fillConvexPoly(mask, np.int32(triangle_intr_crop), (255, 255, 255))
        #mask inversion
        maskinv = 255 - mask
        
        #apply affine transformation to the pixels within cropped regions
        #use cv2.BORDER_REFLECT to fill empty pixels when stitching triangles together
        warp_1 = cv2.warpAffine(imgrect1, affine1, (tri_ri[2], tri_ri[3]), borderMode = cv2.BORDER_REFLECT)
        warp_2 = cv2.warpAffine(imgrect2, affine2, (tri_ri[2], tri_ri[3]), borderMode = cv2.BORDER_REFLECT)
        
        #apply triangle mask to intermediate region to remove pixels outside of triangle
        newimage1 = cv2.bitwise_and(warp_1,warp_1,mask = mask)
        newimage2 = cv2.bitwise_and(warp_2,warp_2,mask = mask)

        #apply alpha blending to image 1 and image 2
        img_intr = cv2.addWeighted(newimage1, 1-k, newimage2, k, 0.0)

        #apply mask to output image to clear triangular region to prevent pixel overlapping
        img3[tri_ri[1]:tri_ri[1]+tri_ri[3], tri_ri[0]:tri_ri[0]+tri_ri[2]] = cv2.bitwise_and(img3[tri_ri[1]:tri_ri[1]+tri_ri[3], tri_ri[0]:tri_ri[0]+tri_ri[2]], img3[tri_ri[1]:tri_ri[1]+tri_ri[3], tri_ri[0]:tri_ri[0]+tri_ri[2]], mask = maskinv)
        
        #add blended and morphed triangle to output image
        img3[tri_ri[1]:tri_ri[1]+tri_ri[3], tri_ri[0]:tri_ri[0]+tri_ri[2]] = img3[tri_ri[1]:tri_ri[1]+tri_ri[3], tri_ri[0]:tri_ri[0]+tri_ri[2]] + img_intr

def create_frames(frames, triangle_list, img1, img2):

    img3 = np.zeros((height,width,3)).astype(np.uint8)
    k = 0
    for f in range(frames):
        morph_triangles(triangle_list, img1, img2, img3, k)
        framefile = "frame/morph" + str(f) + ".jpg"
        cv2.imwrite(framefile, img3)
        k += 1/frames

def morph_faces(filename1, filename2):
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    images = []
    lm_list = []
    triangle_list = []
    images.append(img1)
    images.append(img2)
    find_landmarks(images, lm_list)
    landmark_triangulation(images, lm_list, triangle_list)
    create_frames(40, triangle_list, img1, img2)


if __name__ == '__main__':
    filename1 = "was2.jpg"
    filename2 = "bor.jpg"
    morph_faces(filename1, filename2)