**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_imgs/undist_img.PNG "Undistorted"
[image2]: ./output_images/undistorted_example.jpg "Road Transformed"
[image3]: ./output_images/binary_image.jpg "Binary Example"
[image4]: ./output_images/warped_example.jpg "Warp Example"
[image5]: ./output_images/warped_binary_example.jpg "Binary Warp Example"
[image6]: ./output_images/lane_detect.jpg "Finding lane"
[image7]: ./output_images/final_image.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Here is where the camera calibration was made, a necessary step to correct some distortions made on the image. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function inside my cal_undistort function as shown below:

```python
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
```

And I obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

To undistort the image above, it was simple by using the same function declared above, cal_undistort, and passing the arguments of the img and obj points obtained by the findChessBoardCorners along with the image desired to be unsdistorted.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Here I have tested a lot of different thresholds values to try the best one and also color thresholds for the S channel of the HLS and the R channel of the RGB. I found out that instead of the gray image for the filters like sobel and magnitude, the best was to pass the S and R binary image combined. This is shown in the Section Color and Thresholds of the Project Notebook. To show a part of it, here is a snippet:

Combining the channels for the filters:
```python
a1 = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,2]
a2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[:,:,0]
added = cv2.addWeighted(a1, 0.5, a2, 0.5, 0.0)
image = added.copy()
```

Here's an example of my output for this step:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in the Section named Warping Image of the IPython notebook.  The `warp()` function takes as inputs an image (`img`), and inside the function I wrote down the source (`src`) and destination (`dst`) points. The source and destination points where chosen after several trials and are hardcoded in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 50, img_size[1] / 2 + 100],
    [((img_size[0] / 6) + 55), img_size[1]],
    [(img_size[0] * 5 / 6) + 20, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. This was easily achieved with the help of OpenCV function `cv2.getPerspectiveTransform`.

![alt text][image4]

And also here an example of the binary warped image:
![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For detecting the lanes, I followed the steps from the lessosn, where we  get the histogram of the bottom part of the image to get the argmax of the tow sides where more white pixels appear because of the lane. With that, a sliding window was created to go throught the image within a margin to get the pixels index. With good indices concatenated, for the left and right lanes, it was able to get the positions and with that, fit a second order polynomial with the help of numpy `polyfit()`. It is all written in the Section Detecting Lanes of the jupyter notebook, and the result of the function is:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

For the curvature I created the function below, which uses the function described on lesson Measuring Curvature II, on real world, where we applies a transformation from pixels to meters. It is also in the same function where was calculated the vehicle distance from center, where we can add both curved lanes to get the center of them and subtract from the middle of the frame (640). It is all in the section Calculating the curvature.

```python
def measure_curvature(warped_im, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    ploty = np.linspace(0, warped_im.shape[0]-1, warped_im.shape[0])
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    
    ## Implement the calculation of the left line here
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    
    ## Implement the calculation of the right line here
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    # As per commented by Francesco in the Mentors Help, here we can calculate the vehicle offset
    lane_center = ((left_fit[0]+right_fit[0])*warped_im.shape[0]**2+(left_fit[1]+right_fit[1])*warped_im.shape[0]+ left_fit[2] + right_fit[2])//2
    # we assume the camera is centered in the car
    car_center = warped_im.shape[1]/2
    center_offset = (lane_center - car_center) * xm_per_pix
    
    return left_curverad, right_curverad, center_offset
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in section Warp back the result and final image, where I used the function warpPerspective from OpenCV with the inverse matriz transform, and wrote on the image the curvature and car position from center. Here is an example of my result on a test image:

![alt text][image7]

#### 7. Videos

I saved the project and challenge videos, with the names output on them. Where we have the project_video_output.mp4 and challenge_video_output.mp4, which are on the main folder.

#### 8. Discussion

It was really interesting to develop and study all that was talked about here on the advanced line finding. But it was a lot, I mean a lot of work, with all the tests and tweaks needed, and to in the end, not have a very good output, specially with shadows.

I have tested and tried a lot of combinations regarding the thresholds of the sobel and magnitude filters, as well as the color thresholding. Tried HSV, HLS, LAB and RGB, and still struggled with shadows and differences on the paviment. In the end, I believed that even not going well on shadows of the challenge video, it could find the lanes.

To get a really good threshold on the lanes, was a mandatory point to get the radius of curvature right. I saw that if dont have enough pixels, it wouldnt find the correct curvature, causing to miscalculations, but due to not getting all the pixel from the lanes and fitting a good polynomial.

In the end I was able to perform well on the project and OK in the challenge, apart from the shadow from the bridge. I have tried with a custom video, filmed from my cellphone, but there where too many shadows and the same filters did not work well, so I am making it all again but for different resolution and pixels intensity. This shows that the same algorithm and threshold used here wont be good for a different weather condition and time of the day, even using the same camera.