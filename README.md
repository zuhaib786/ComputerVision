This repository contains soliutions to the assignments in the course COL780 Computer Vision
# Assignment 1 - Background Subtraction KDE
- Implementation of Background subtraction using Kernel Density Estimation(KDE). The implementation is relatively slow than that of openCV method, because we do not use lookup tables for calculating the kernel function values
- The detected objects are then enclosed in bounding boxes. A bounding box is considered a valid bounding box if the fraction of foreground pixels in that box is greater than some threshold. We use integral images to get the sum of foreground pixels within a bounding box and non-maxinmal supression to get the valid boxes only. Non maximal supression is performed as follows
    - Sort the bounding boxes in order of the fraction of foreground pixels in the bounding box in descending order
    - Take the first bounding box in the list as valid bounding box and remove all bounding boxes with IoU with the current bounding box greater than some threshold
    - Repeat steps 2 and 3 until all bounding boxes are either removed or have been selected as valid bounding boxes
- Finally a bounding box is drawn around the obtained bounding boxes and all the portion outside of bounding box is considered as part of background

# Assignment 2 - Panorama Generation
- In this assignment we implement ``Harris Corner Detector`` and then find Affine matrix between two views. Using the affine matrix, we generate a panorama
- Sobel filters are used to get the derivatives and 8 way optima is used to perfom non-maximal supression in a window of size $3\times3$ to get the appropriate corners
- Among the corners obtained top $K$ are then selected.
- Finally matching of corner points between two images is performed based on proximity and Sum of squared errors
- Using this matching beween the corner points of the two images affine/homography matrix is obtained and then that affine/homography matrix is used for stitching the two images using ``cv2.warpAffine``/``cv2.warpPerspective``
- The generated panorama is then trimmed to remove the extra portion.
- We assume that images move from left to right
# Assignment 3 - Camera Callibration and Image Augmentation
- Zhangs Algorithm is implemented to find the intrinsic parameters of the camera matrix
- Then the Extrinsic parametrs are obtained from the homography matrix
- Using extrinsic and intrinsic parameters projection matrix is obtained and then that projection matrix is used to project a cube onto the image
- This leads to generation of an augmented image
