from PIL import Image, ImageDraw
import cv2
import imageio
import numpy as np
import random
import time

def main():

    # edge detection parameters
    edge_threshold1 = 500
    edge_threshold2 = 1000
    
    # width for Gauss distribution of velocities
    sigma_vel = 0.01

    # constant for harmonic force
    kf = -0.1

    # coefficient for friction force
    frix = -0.5

    # timestep
    delta_t = 0.2

    # open video input stream
    capture_stream = cv2.VideoCapture(0)
    
    # get size of video input
    xmax = capture_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    ymax = capture_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    xmax_px = int(xmax)
    ymax_px = int(ymax)
    print("xmax / pixel ", xmax_px)
    print("ymax / pixel ", ymax_px)

    # particles reservoir size
    dim_part = int(xmax * ymax/30)
    print("max number of particles ", dim_part)
        
    # random distribution of particles
    x_part, y_part = random_pos_part(dim_part, xmax, ymax)

    # Gauss distribution of velocities
    vx_part, vy_part = gauss_vel_part(dim_part, sigma_vel)

    # initialize test energy counts and list of image files
    en_test = 0.
    count_loop = 0
    count_img = 0
    filenames = []

    while(count_loop < 4):
        
        # initialize forces
        force_x = np.zeros(dim_part, dtype=float, order='C')
        force_y = np.zeros(dim_part, dtype=float, order='C')

        if en_test < 0.5:
            time.sleep(2.)

            # detect edges in video 
            edges = find_edges(capture_stream, edge_threshold1, edge_threshold2)

            # edges define anchor points
            y_anchpts, x_anchpts = np.where(edges==1)
            dim_anchpts = x_anchpts.shape[0]
            print("number of anchor points and active particles ", dim_anchpts)
            count_loop += 1
            print("Loop {}".format(count_loop))

        # loop over particles

        #energy
        en = 0.
        for i in range(0,dim_anchpts):

            # update positions
            aux = x_part[i] + delta_t*vx_part[i]
            x_part[i] = aux
            aux = y_part[i] + delta_t*vy_part[i]
            y_part[i] = aux

            # keep particles in domain
            if x_part[i] > xmax:
                x_part[i] = x_anchpts[i]
            if x_part[i] < 0.:
                x_part[i] = x_anchpts[i]
            if y_part[i] > ymax:
                y_part[i] = y_anchpts[i]
            if x_part[i] < 0.:
                y_part[i] = y_anchpts[i]                
            
            # calculate harmonic force and energy
            distx = x_part[i] - x_anchpts[i]
            disty = y_part[i] - y_anchpts[i]
            force_x[i] = kf*distx
            force_y[i] = kf*disty
            en = en + distx*distx + disty*disty

            # add friction force
            aux = force_x[i] + frix*vx_part[i]
            force_x[i] = aux
            aux = force_y[i] + frix*vy_part[i]
            force_y[i] = aux
            
            # update velocities
            aux = vx_part[i] + delta_t*force_x[i]
            vx_part[i] = aux
            aux = vy_part[i] + delta_t*force_y[i]
            vy_part[i] = aux

        # free particles move randomly
        for j in range(dim_anchpts+1,dim_part):
            x_part[j] = random.uniform(0.,xmax)
            y_part[j] = random.uniform(0.,ymax)

        # screen output
        vis=np.zeros((ymax_px,xmax_px), dtype=np.float32)
        vis[y_part.astype(int), x_part.astype(int)] = 1 
        cv2.imshow('Gotcha!', vis)

        # store image
        count_img += 1
        cv2.imwrite('images/test_{}.jpg'.format(count_img),vis*255,[cv2.IMWRITE_JPEG_QUALITY, 50])
        filenames.append('images/test_{}.jpg'.format(count_img))
        
        cv2.waitKey(1)

        # update test energy
        en_test = en / float(dim_anchpts)
           
    # release webcam feed
    capture_stream.release()
    cv2.destroyAllWindows()

    # assemble images in animated gif
    with imageio.get_writer('images/movie.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


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

    # get frame from webcam
    _, frame = capture_stream.read()

    # convert frame to grey scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect edges with John Canny algorithm
    edges = cv2.Canny(gray, edge_threshold1, edge_threshold2, apertureSize=5)
    
    # convert arrays that contains 0s and 1s
    edges = np.where(edges != 0, 1, 0)
    print("Gotcha!")
    return edges


def get_video_input_size(capture_stream):

    # get frame from webcam
    _, frame = capture_stream.read()

    return frame.shape


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
    
# Gauss velocity
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