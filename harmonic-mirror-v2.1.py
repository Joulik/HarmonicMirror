import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import time

#Version 2.1 is just version 2 that was cleaned
#Credit to acfogarty for helping me to start with python and edge detection 

def main():

    ###### PARAMETERS
    #----- NUMBER OF STEPS BETWEEN TWO SNAPSHOTS
    #snap_period = 100
    #----- EDGE DETECTION PARAMETERS
    edge_threshold1 = 500
    edge_threshold2 = 1000
    #----- SIZE OF SQUARE-SUBARRAY FOR EDGES SIMPLIFICATION MASK
    edges_mask_step = 1
    #----- WIDTH FOR GAUSS DISTRIBUTION OF VELOCITIES
    sigma_vel = 0.01
    #----- CONSTANT FOR HARMONIC FORCE
    kf = -0.1
    #----- COEFFICIENT FOR FRICTION FORCE
    frix = -0.5
    #----- TIMESTEP
    delta_t = 0.2

    ###### OPEN VIDEO INPUT STREAM
    capture_stream = cv2.VideoCapture(0)
    
    ###### GET SIZE OF VIDEO INPUT
    xmax = capture_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    ymax = capture_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    xmax_px = int(xmax)
    ymax_px = int(ymax)
    print("xmax / pixel ", xmax_px)
    print("ymax / pixel ", ymax_px)

    ###### PARTICLES RESERVOIR SIZE
    dim_part = int(xmax * ymax/15)
    print("max number of particles ", dim_part)
        
    #----- RANDOM DISTRIBUTION OF PARTICLES
    x_part, y_part = random_pos_part(dim_part, xmax, ymax)

    #----- GAUSS DISTRIBUTION OF VELOCITIES
    vx_part, vy_part = gauss_vel_part(dim_part, sigma_vel)

    #----- INITIALIZE TEST ENERGY
    en_test = 0.
    
    while(True):
        
        #----- INITIALIZE FORCES
        force_x = np.zeros(dim_part, dtype=float, order='C')
        force_y = np.zeros(dim_part, dtype=float, order='C')

        if en_test < 0.5:
            time.sleep(2.)
            #----- DETECT EDGES IN VIDEO 
            edges = find_edges(capture_stream, edge_threshold1, edge_threshold2)
            #print("Gotcha!",str(step),str(en_test))

            #----- EDGES DEFINE ANCHOR POINTS
            y_anchpts, x_anchpts = np.where(edges==1)
            dim_anchpts = x_anchpts.shape[0]
            print("number of anchor points and active particles ", dim_anchpts)
            
        #----- LOOP OVER PARTICLES
        en = 0. # energy
        for i in range(0,dim_anchpts):

            #----- UPDATE POSITIONS
            aux = x_part[i] + delta_t*vx_part[i]
            x_part[i] = aux
            aux = y_part[i] + delta_t*vy_part[i]
            y_part[i] = aux

            #----- KEEP PARTICLES IN DOMAIN
            if x_part[i] > xmax:
                x_part[i] = x_anchpts[i]
            if x_part[i] < 0.:
                x_part[i] = x_anchpts[i]
            if y_part[i] > ymax:
                y_part[i] = y_anchpts[i]
            if x_part[i] < 0.:
                y_part[i] = y_anchpts[i]                
            
            #----- CALCULATE HARMONIC FORCE AND ENERGY
            distx = x_part[i] - x_anchpts[i]
            disty = y_part[i] - y_anchpts[i]
            force_x[i] = kf*distx
            force_y[i] = kf*disty
            en = en + distx*distx + disty*disty

            #----- ADD FRICTION FORCE
            aux = force_x[i] + frix*vx_part[i]
            force_x[i] = aux
            aux = force_y[i] + frix*vy_part[i]
            force_y[i] = aux
            
            #----- UPDATE VELOCITIES
            aux = vx_part[i] + delta_t*force_x[i]
            vx_part[i] = aux
            aux = vy_part[i] + delta_t*force_y[i]
            vy_part[i] = aux

        #----- FREE PARTTICLES MOVE RANDOMLY
        for j in range(dim_anchpts+1,dim_part):
            x_part[j] = random.uniform(0.,xmax)
            y_part[j] = random.uniform(0.,ymax)

        #----- SCREEN OUTPUT
        vis=np.zeros((ymax_px,xmax_px), dtype=np.float32)
        vis[y_part.astype(int), x_part.astype(int)] = 1 
        cv2.imshow('Gnouglouf', vis)
        cv2.waitKey(1)

        #----- UPDATE TEST ENERGY
        en_test = en / float(dim_anchpts)
           
    #----- REALEASE WEBCAM FEED
    capture_stream.release()
    cv2.destroyAllWindows()

##### FIND_EDGES #####    
def find_edges(capture_stream, edge_threshold1, edge_threshold2):
    '''
    find edges in a frame from the video stream

    Input:
        capture_stream: open video capture stream object
        edge_threshold1, edge_threshold2 (int): adjustable detection thresholds 
    Returns:
        2D numpy array with dimensions n_pixels_height * n_pixels_width 
        containing 1 where there is an edge and 0 otherwise
    '''

    #----- GET FRAME FROM WEBCAM
    _, frame = capture_stream.read()
    #----- CONVERT FRAME TO GREYSCALE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #----- DETECT EDGES WITH John Canny ALGORITHM
    edges = cv2.Canny(gray, edge_threshold1, edge_threshold2, apertureSize=5)
    #----- CONVERT TO ARRAY THAT CONTAINS 0's AND 1's
    edges = np.where(edges != 0, 1, 0)
    print("Gotcha!")
    return edges

##### GET VIDEO INPUT SIZE #####
def get_video_input_size(capture_stream):

    # get frame from webcam
    _, frame = capture_stream.read()

    return frame.shape

##### RANDOM POS PART #####
def random_pos_part(dim_part, xmax, ymax):
    '''
    generate random x and y coordinates for particles
    input:
        dim_part (int): length of output arrays
        xmax, ymax : define domain in xy plane
    '''

    x_part = np.random.uniform(low=0., high=xmax, size=dim_part)
    y_part = np.random.uniform(low=0., high=ymax, size=dim_part)

    return x_part,y_part
    
##### GAUSS VEL PART #####
def gauss_vel_part(dim_part,sigma_vel):
    '''
    generate gauss distribution of particles velocities
    input:
        dim_part (int): length of output arrays
    '''

    vx_part = np.random.normal(0.0, sigma_vel, size=dim_part)
    vy_part = np.random.normal(0.0, sigma_vel, size=dim_part)

    return vx_part, vy_part

if __name__ == '__main__':
    main()
