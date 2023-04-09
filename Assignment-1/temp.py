import numpy as np
def foregroundAggregation(integralImages, prediction):
    y, x = prediction.shape
    prediction = np.ones(prediction.shape)
    ans = np.zeros((y, x))
    ans[:, :] = prediction
    for i in range(9, y):
        for j in range(9, x):
            # if prediction[i - 4, j - 4] == 1:
            #     sm = np.sum(prediction[i - 9:i + 1, j- 9: j + 1].ravel())
            #     if sm < 20:
            #         ans[i-4, j-4] = 0
            ans[i - 4, j- 4] = 1
    return ans

    return ans