# Method used
Implemented Background detection with Kernel density estimation with exponetially decaying weights to the past $N$-values<br>
The probability density function estimate is given by <br>
$$f(x) = \displaystyle \frac{\displaystyle \sum_{i = 0}^{n - 1}\alpha^i K(x, x_i, \sigma)}{\displaystyle \sum_{i = 1}^{n - 1}\alpha^i}$$
We predict background when $f(x_t) >\text{threshold}$ for a chosen threshold<br>
The estimation of the kernel width $\sigma$ is given by $\displaystyle \frac{m}{0.68\sqrt{2}}$ where $m$ is the median of $|x_i - x_{i - 1}|, 2\leq i\leq n$<br>
# Foreground Pixel Aggregation
To suppress false detections we use foreground pixel aggregation method where for a pixel position $(i, j)$ predicted as foreground we aggregate the pixels in the rectangular region $i- 5\leq x\leq i + 5, j-5\leq y\leq j + 5$. If the count of pixels does not exceed some chosen threshold, then the pixel position is considered as part of noise and removed.
# Bounding Box detection
We set the probability of an object being present in the bounding box be given by the ratio of the number of pixels predicted as foreground and the total number of pixels in the bounding box. We use non-maximal suppression to predict the appropriate bounding box capping the bounding box overlap at some chosen threshold
# Comparison
|KDE with exponentially decaying weights| GMM with exponetially decaying weights|
|---------------------------------------|---------------------------------------|
|Model responds quickly to the changes in background| Model is slow to respond to changes in background compared to that of KDE. Sometimes a mirage of a moving object is created because of that|
|Relatively noisier as more false detection occur | Less noisy compared to that of KDE|
|Model performs poorly when a moving objects stands still for some time as it amalgamates it to the background. This can be seen for the case of "Hall And Monitor" data and "Candela_m1.10" data| Model relatively robust when a moving objects stands still for some time|

# Failure cases of the method
1. The kernel width $\sigma$ plays a very important role in the prediction accuracy by the model. The calculation for the value $\displaystyle \exp\left( - \displaystyle \frac{(x - \mu)^2}{\sigma^2}\right)$ overflows very quickly if the variance is small and hence a lot of false detections occur because of that.<br>
__Hence on a still scene a lot false detections can occur because of this method.__<br>
2. This model does not work well for detecting objects with varying speeds because if an object is slow, it gets amalgamated with the background and if the adaptation(learning) rate is slow, then the model does not adapt quickly to the background changes.

# Links to videos
[Link](https://csciitd-my.sharepoint.com/:f:/g/personal/mt6180798_iitd_ac_in/EuNHBSCUFoBGne-TcmgAvX0BTFb4WtbbJCE3Nm7-i6GoHg?e=OxN0tX)
