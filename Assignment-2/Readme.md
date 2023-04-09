__Submission By__: _Zuhaib Ul Zamann_<br>
__Entry Number__: 2018MT60798
# Approach
## Part 1: Harris corner detector
File: Harris.py<br>
Class HarriCornerDetector<br>
Fields:
- __Kernels__: Dictionary of sobel filters
```python
Kernels['x']: #Filter for derivative across x-coordinate
Kernels['y']: #Filter for derivative across y-coordinate
```

Methods:
- __Convolve2D__: 2D-Convolution operation with padding set as same, so that the input and output image size is same
- __Smoothen__: Applying gaussian blur to smoothen the picture
- __HarrisMatrix__: Calculates harris matrix values given by $det(H) - 0.05 tr(H)^2$. Elements of harris matrix are calculated by taken gaussian weighted sum of each of the terms involved
- __nonMaxSupression__: Perform nonMaxSupression in a window of windowSize. 8-Way local optima is found
- __SelectTopK__: After non-max Supression select the first K points with maximum harris corner value
- __getCorners__: Method for getting the final keypoints(corners)

File: Matching.py
Class Matching<br>
Fields:
- __detector__: HarrisCornerDetector. Used for finding Key points

Methods:
- __findNearestMatches__: matches keypoints based on the value of $0.1\times SSE + \text{Manhat(pixel coordinates of points)}$, where $\text{Manhat(x,y)}$ is the manhattan distance between x and y given by $\displaystyle\sum_{i = 1}^n|x_i - y_i|$

- __match__:- Wrapper function over ``Matching.findNearestMatches``
- __findAffine__: Wrapper function over cv2 library function ``cv2.findHomography``. Uses ``Matching.match`` to get the matches and then calls ``cv2.findHomography``
- __trim__: Function to crop the image to get rid of the extra 0s
- __findHomograpgies__: calls ``findAffine`` over adjacent images to get the homographies between adjacent images
- __generatePanoroma__: Takes folder address, loads images using ``os.listdir`` and ``cv2.imread`` and then calls ``self.detector`` to get the key points and  ``Matching.findhomographies`` to get the homographies. Using ``cv2.warpPerspective``, applies homography on right images and overlays left image(Because the homography calculates the view in coordinates of left image). Saves the final panorama at the folder address with name as ``pan.jpg``
Image is resized to (400, 400) to reduce computational cost

File: AffineModel.py<br>
Class AffineModel<br>
Fields:
- __detector__: HarriscCornerDetector. Used to calculate keypoints
- __match__: Matching, used to calculate matching between keypoints
- __createMatrix__: Given matches as $((x_1, y_1)\rightarrow (x_1', y_1'), (x_2, y_2)\rightarrow (x_2', y_2'), \ldots, (x_n', y_n'))$ creates the matrix.
$\begin{bmatrix}x_1 & y_1& 1&0& 0&0\\
                0 & 0& 0 & x_1& y_1& 1\\
                x_2 & y_2 & 1& 0& 0& 0\\
                0 & 0 & 0 & x_2 & y_2 & 1\\
                \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
                x_n & y_n & 1& 0 & 0 & 0\\
                0 & 0& 0 & x_n& y_n&1
\end{bmatrix}$.<br>
We only pass the source points and denote the matrix by $A$
- __createLoad__: Given matches as $((x_1, y_1)\rightarrow (x_1', y_1'), (x_2, y_2)\rightarrow (x_2', y_2'), \ldots, (x_n', y_n'))$ creates the matrix.
$\begin{bmatrix}x_1'\\y_1'\\x_2'\\y_2'\\ \vdots\\x_n'\\y_n' 
\end{bmatrix}$<br>We only pass the destination points and denote the matrix by $b$
- __estimateParams__: Linear regression estimate for the equation $At = b$, where $t = \begin{bmatrix}
a \\b\\c\\d\\e\\f
\end{bmatrix}$ and the affine transform is then given by
$\begin{bmatrix}a & b & c\\ d & e & f\end{bmatrix}$
- __builtinEstimator__: Use ``cv2.estimateAffine2D`` to estimate the affine matrix params
- __getMatches__: Same as ``Matching.match`` function
- __getAffine__: similar to ``Matching.findAffine``. Instead of homography, it finds the affine matrix
- __getAffines__: Similar to ``Matching.findHomographies``
- __generatePanorama__: Similar to ``Matching.generatePanorama``. Uses ``AffineModel.getAffines`` in place of ``Matching.findHomographies`` and ``cv2.warpAffine`` instead of ``cv2.warpPerspective`` 

# Requirements
opencv-python==4.7.0<br>
matplotlib==3.6.2<br>
numpy==1.23.4<br>
tqdm==4.64.1<br>
scipy==1.10.0
# How to run
In the file affine Model change the line at last
```python
A.generatePanorama('NewData/5')
```
to 
```python
A.generatePanorama(<yourfolderAddress>)
```
Make sure that if the frames are sorted in lexicographical order, then the images should run from left to right<br>
Now from command line run the script ``AffineModel.py`` and the output will be saved as __pan.jpg__ in the folder from which data is loaded
# Results
[One drive Link](https://csciitd-my.sharepoint.com/:f:/g/personal/mt6180798_iitd_ac_in/Eh0gxZlVFjhGgWGWWylqZ2cBiOF_MfuYDgQvO_wvssRktQ?e=vCYT2C)