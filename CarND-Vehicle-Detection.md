# **Vehicle Detection**

![alt text][image1]

---


**Vehicle Detection Project**

### Introduction

In this project, we will build a pipeline to detect and track vehicles using color and gradient features and a support vector machine classifier.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./resources/preface.png "Preface"
[image2]: ./resources/original_images.png "Original"
[image3]: ./resources/cars.png "Cars"
[image4]: ./resources/notcars.png "Not Cars"
[image5]: ./resources/hog.png "HOG Features"
[image6]: ./resources/sliding_windows.png "Sliding Windows"
[image7]: ./resources/bboxes.png "Bounding Boxes"
[image8]: ./resources/heat.png "Heatmaps"
[image9]: ./resources/labels.png "Labels"
[image10]: ./resources/pout.png "Pipeline Output"

### Download the data

I download the dataset labeled as [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) for the project to extract features, then train a classifier and ultimately track vehicles in a video stream. 

### Data Set Summary & Exploration

**Summary dataset**

Here is the summary statistics of the vehicle and non-vehicle data set:

* Number of car images: 8792
* Number of notcar images: 8968
* Image shape: (64, 64, 3)
* Total number of samples: 17760
* The size of training set is 14208
* The size of the test set is 3552





**Original images**

The test images have the shape (720, 1280, 3), meaning a height of 720, a width of 1280, and 3 RGB channels. 

![alt text][image2]

**Dataset split**

I shuffle and split the data into training(80%) and test(20%) datasets.







### Histogram of Oriented Gradients (HOG)

#### Extracted HOG features

The code for this step is contained in the file called `gen_model.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:


Here are the random 20 samples of vehicles
![alt text][image3]

Here are the random 20 samples of non-vehicles
![alt text][image4]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

I extract HOG features from the first channel of test images.

```python
images = glob.glob('test_images/*.jpg')
for idx, fname in enumerate(images):
    image = cv2.imread(fname)
    image = BGR2RGB(image)
    _, car_hog = get_hog_features(image[:,:,0], 9, 8, 8, vis=True, feature_vec=True)
    test_images.append(car_hog)

show_images(test_images)
```

![alt text][image5]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I have been able to achieve >98% accuracy with simple SVC classifiers without doing a lot of experimentation to search for good hyperparameters.

I experment different color spaces and skimage.hog() parameters (orientations, pixels_per_cell, and cells_per_block) to come up with final tuned parameters, Here are my parameters:

```python
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using extracting features for each sample in the training set and supplying these feature vectors to the training algorithm, along with corresponding labels.

The data was properly random shuffled, split into training and testing sets and normalized.

```python
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print("The size of training set is {}".format(len(X_train)))
    print("The size of the test set is {}".format(len(X_test)))

    X_train, y_train = sklearn.utils.shuffle(X_train, y_train)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
```

I am able to achive 99.4% test accuracy on the dataset.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Because of searching the total number of windows can make the algorithm run slower, i decided to restrict the search to the only areas of the images where vehicles might appear.

```python
window_scale = (1.0, 1.25, 2)
x_start_stop = [[624, 1024], [400, 1280], [384, 1280]]
y_start_stop = [[400, 480], [375, 520], [400, 656]]
xy_window = (80, 80)
xy_overlap = (0.75, 0.75)
color_values = [(0,0,255), (0,255,0), (255,0,0)]

test_images = []
images = glob.glob('test_images/*.jpg')
for idx, fname in enumerate(images):
    image = cv2.imread(fname)
    image = BGR2RGB(image)
    for i, scale in enumerate(window_scale):
        windows = slide_window(image, x_start_stop=x_start_stop[i], y_start_stop=y_start_stop[i],
                                    xy_window=[int(dim*window_scale[i]) for dim in xy_window], xy_overlap=xy_overlap)
        image = draw_boxes(image, windows, color_values[i])
    test_images.append(image)

show_images(test_images)
```

![alt text][image6]



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on six scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image7]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)


[![P5](https://img.youtube.com/vi/UvSpBcvKiRc/0.jpg)](https://www.youtube.com/watch?v=UvSpBcvKiRc "Vehicle Detection")



#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image8]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image9]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image10]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


The classifier detect the cars nearby very well but very poor for detecting the faraway small cars, even though the SMV test accuracy is a good 0.994, I try to work on either changing lower scale to something little higher or else try playing with ystart and ystop for that scale, and also work on the heatmap thresholding which can eliminate smaller detections where the boxes don’t fit quite right, after experiment, i get an ideal output with very few false positives.

I will try to use a Deep Learning based approach instead of the HOG approach for vehicle detection, i.e. SSD or YOLO.
