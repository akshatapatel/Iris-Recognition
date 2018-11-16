# Iris-Recognition
Iris recognition using applied machine learning on CASIA iris images dataset

# IrisLocalization.py


## Variables:-
images=list of all images
target=contains the colored images
boundary=list that contains final images with inner and outer boundaries of the iris,initially empty
centers=list that contains pupil centers of all images,initially empty
draw_img=image over which circles will be drawn
a = (crop_center_x,crop_center_y): Pupil center


The function IrisLocalization(images) does the following:
1. It uses a Bilateral filter to remove the noise by blurring the image
2. We project the image coordinates in the horizontal and vertical directions, and find the minimum(as the minimum would be a dark region of the pupil) to find the approximate center of the pupil.
3. We next use this approximate center to binarize a 120 x 120 space around the pupil, as defined in the paper, to re-estimate the pupil center.
4. We perform Canny edge detection on a masked image to get only a few edges around the pupil. If the image was not masked we would get a lot of edges of the eyelashes,etc.
5. We then implement Hough transformation on the edged image to detect all possible circles in the image. We take the Hough circle that has center closest to the pupil center found as the pupil boundary.
6. The outer boundary is then drawn by adding 53 to the radius of the inner circle.
7. The list “boundary” stores all the images with boundaries drawn, while the list “centers” stores the center coordinates.


# IrisNormalization.py

## Variables:-
normalized=initially empty, stores all normalized images 

cent=counter variable that loops through all center values

Center_x,Center_y=coordinates of pupil center

nsamples=defines number of equally spaced intervals

polar=array that will store the normalized image


We sequentially load each image from the list boundary returned by the previous function and initialize an empty list to store the normalized images
1. In order to project the polar coordinates onto the cartesian plane, we only need to focus on the region between the boundaries.
2. We define an equally spaced interval over which the for loop iterates to convert polar coordinates to cartesian coordinates, using x=rcos(theta) and y=rsin(theta)
3. We resize the image to a rectangular 64x512 sized image


# ImageEnhancement.py

## Variables:- 
normalized=list of normalized images


In this function, we enhance the image using Histogram Equalization to increase the contrast of the image for better feature extraction.


# FeatureExtraction.py

## Variables:- 
Filter1: 1st filtered image
Filter2: 2nd filtered image
Img_roi: 48*512 rectangular image of interest
Feature_vector: the feature vector found by combining the means and standard deviation for each of the two channels


The functions ‘m’ and ‘gabor’ help in calculating the spatial filter defined in the paper


Code:
def m(x ,y, f):
    val = np.cos(2*np.pi*f*math.sqrt(x **2 + y**2))
    return val


def gabor(x, y, dx, dy, f):
    gb = (1/(2*math.pi*dx*dy))*np.exp(-0.5*(x**2 / dx**2 + y**2 / dy**2)) * m(x, y, f)
    return gb


def spatial(f,dx,dy):
    sfilter=np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            sfilter[i,j]=gabor((-4+j),(-4+i),dx,dy,f)
    return sfilter


The function spatial takes f, dx and dy as parameters which are defined in the paper.
It creates a 8x8 block which we run over the normalized image to extract features.

As we have eyelashes and other occlusions present in normalized images, we run this 8x8 block over a normalized image of 48*512, which is our region of interest. This helps in further improving the results of matching.


Code: 
filter1=spatial(0.67,3,1.5)
 filter2=spatial(0.67,4,1.5)
 feature_vector=[]
 for i in range(len(enhanced)):
        img=enhanced[i]
        img_roi=img[:48,:]
        filtered1=cv2.filter2D(src=img_roi, kernel=filter1, ddepth=-1)
        filtered2=cv2.filter2D(src=img_roi, kernel=filter2, ddepth=-1)


The function above defines this Region of interest and creates two channels: filter1 and filter 2, which are then convolved with our RoI to get two filtered images.


Code snippet for get_vec():
for i in range(6):
            for j in range(64):
 start_height = i*8
 end_height = start_height+8
 start_wid = j*8
 end_wid = start_wid+8
 grid1 = convolvedtrain1[start_height:end_height, start_wid:end_wid]
 grid2 = convolvedtrain2[start_height:end_height, start_wid:end_wid]


These filtered images are then used to get our feature vector using the function ‘get_vec()’
This function calculates the mean and standard deviation values grid by grid, where each grid is a 8x8 block. The calculated values are then appended sequentially to the feature vector, which is of size 1536 (6x64x4).


# IrisMatching.py
This py file matches our testing and training feature vectors (with and without dimensionality reduction).


## Function dim_reduction:
## Variables:-
feature_vector_train: list of feature vectors of the train images
feature_vector_test: list of feature vectors of the test images
components: list of different dimensions that the feature vector has to be reduced to
y_train :contains the labels of the classes of the training feature vectors for 108 eyes.

## Function IrisMatching
## Variables:-
Feature_vector_train,feature_vector_test: input feature vectors
red_train,red_test: reduced feature vector for training and testing (same as original in case of no reduction)
components: input of the number of reduced dimensions 
index_L1,index_L2,index_cosine: values of matching indexes with minimum distance from a test image (size 432)
sumL1, sumL2, cosinedist: actual values of all 3 distance parameters
match, count: used to compare the matched index to the actual index to determine matching
match_L1,match_L2 and match_cosine: stores 0/1 if incorrectly or correctly matched
match_cosine_ROC: stores 0/1 by matching it using threshold values explained below
Thresh: threshold values for ROC


The function dim_reduction(feature_vector_train,feature_vector_test,components) does the following:
It fits our LDA model to training data
Then it transforms both training and testing feature vectors to the number of components specified in the parameters (max 107)


The function IrisMatching(feature_vector_train,feature_vector_test,components,flag) does the following:
1. If flag==1 then it performs matching on the original feature vector of size 1536.
If flag==0 then it performs matching on the reduced feature vector of size provided by the components parameter.
2. To perform matching, we calculate the L1, L2 and cosine distances as defined in the paper for every test image against all training samples. Then, the minimum value for all 3 distances is taken as the matched image and its index is stored.
3. Next, if the matched index for each image is correct (i.e, the matched image is from the same folder as our test image) then it is considered as the correct match, and 1 is appended to our “match” list. The same is repeated for all three distances to get match_L1,match_L2 and match_cosine and they are returned
4. We also calculate ROC matching here. If the cosine distance is less than the threshold, then it is 1 (accepted) otherwise it is 0 (rejected). This is stored for all 3 threshold values and returned


# PerformanceEvaluation.py

## Variables:-
correct_L1,correct_L2,correct_cosine: store the elements that are correctly matched i.e, have a value of 1
crr_L1,crr_L2 and crr_cosine: stores the CRR value for L1, L2 and cosine distance


This py file calculates the correct recognition rates for our code.
The function PerformanceEvaluation(match_L1,match_L2,match_cosine) does the following:
   1. The value of the correct correction rate would be given by the count of the eyes that are correctly matched divided by the count of the total number of eyes. This is calculated by dividing the length of correct_L1(as it has only those eyes that are correctly matched) by length of match_L1(as it has all the eyes).
   2. Thus, we get the values of crr_L1,crr_L2 and crr_cosine.


# IrisRecognition.py

This py file is the main file, where we call all the above functions to execute the entire function of iris recognition. 
   1. First, we read and process all training files:
   1. Read all files
   2. Run iris localization on all of them and get the images
   3. On the localized images, run normalization and enhancement to get our enhanced images for feature extraction
   4. Then, run feature extraction to get all the feature vectors
   1. The same steps a-d are followed for the testing data
   2. Once we have our training and testing feature vectors, we run iris matching and performance evaluation on them as:
   1. Get matching values and then CRR scores for 10,40,60,80,90 and 107 components in the reduced feature vector
   2. Use those values to plot the CRR vs Feature Vector dimensions graph
   3. Get matching values and then CRR scores for our full length 1536 feature vector 
   4. Use the 1536 component CRR’s and 107 component CRR’s to plot the table to compare their accuracy
   5. ROC requires the rate of false matches and the rate of false non-matches. False matches are the number of eyes that are matched but are not authorized whereas False non-matches are the number of eyes that are rejected but are authorized.
To calculate ROC, we use the matching_cosine_ROC we got from IrisMatching() and compare it with our actual matching_cosine answer to calculate the FMR and FNMR for all three threshold values.
FMR= no. of images incorrectly accepted / total number of accepted images
FNMR = no. of images incorrectly rejected / total no of rejected images


# IMAGES

This file contains 8 images depicting the step by step output we obtained for localization, normalization and enhancement.

Fig1 depicts the grayscale image of the eye

Fig2 depicts the colored image which is what get stored in the target array in the Localization function

Fig3 depicts the output of Localization, i.e the original image with inner and outer boundaries

Fig4 depicts the Enhanced Normalized 64x512 rectangular image which is used for further feature extraction steps

Fig5 shows the recognition results using features of different dimensionality

Fig6 table that shows the car for L1,L2,cosine fro original and reduced feature vectors

Fig7 shows the ROC curve

Fig8 table that shows ROC fmr and fnmr measures for different thresholds.
