# historical map project

## Progress tracker
- [x] Crop images into small batches for solving memory issue and fast or parallel processing
- [ ] Color Quantization
  - [x] Choose the right color space (we use HSV -> more close to human perception)
  - [x] Color Quantization (3 steps)
      - [x] Mean shift
      - [x] Median cut (option: for reducing the processing time of K-means)
      - [x] K-means
      - [ ] Add spatial information for clustering pixels (pixel coordinates in the images); pixels which are close have higher probability to be considered in the same cluster (TODO)
      - [] Deep neural network detection method (LS-Net)


- [ ] Maps segmentation
    - [x]  Seperate different layers of historical maps
      - [x] Classify layers automatically through the output of the K-means based on the number of ratio between black and white pixels (eg. background layers have the highest number of pixels (value of 0)) <br/>
            Advantage: Do not need any human intervention and simple <br/>
            Disadvantage: Might raise issues when there is more colors in the images <br/>
            Background layers might not belongs the image with highest number of pixels 
      - [x] Users can select layers as they wanted (Manually), for example, background, text and map legend etc. <br/>
            Advantage: maximum filtered out noise pixels <br/>
            Disadvantage: need a lot of clicks from users
    - [ ]  Extract longitude and latitude
      - [x] Hough line transform
        - [x] Inverted images of background are used
        - [x] Use hough transform with selected theta value (90 degree, 0 degree and 180 degree) to detect horizontal and vertical lines from the maps and we measure the percentage of overlapped pixels between detected lines and overlapped pixels of the background<br/>
              Advantage: This method is straight and simple <br/>
              Disadvantage: The hough transform is slow and we need to set variety of threshold for different images to extract the right lines; <br/>
              Broken lines and bad image quality influences the results of line extraction
      - [x] Morphologic method <br/>
            Advantage: Fast<br/>
            Disadvantage: Erode and dilation might connect some unwanted pixels that cause false positive detection <br/>
            The best kernel filter requires to be set manually for different map resources (low generalizibility)
      - [ ] Convolution based line detection method (TODO)
    - [ ]  Detect and remove text from the map 
      - [ ] Text area detection (on going) <br/>
		Issue: The text detector have bad results for detecting our map dataset. <br/>
		Becuase they are trained through english words then French words, a network fine tunning requires to be done
        - [x] EAST text detector
        - [ ] Textbox++ text detector (on going)
      - [ ] Text pixels identification and removal
    - [ ]  Segment homogenous region
    - [ ]  Assign class to each region
        - [ ] Road
        - [ ] rivers
        - [ ] parcels
        - [ ] buildings
        - [ ] sidewalks (in most recent maps)
        - [ ] detect and recognize (read) the text
