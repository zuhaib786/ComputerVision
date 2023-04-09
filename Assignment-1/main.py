from BackgroundSegmentation import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
base_path = 'Test_Dataset_Assignment_1/'
input_dir = base_path + 'Test Dataset'
output_dir = base_path + 'groundtruth'
save_dir = base_path + 'BoundingBox/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
video = Video(input_dir)
# video = cv2.VideoCapture(base_path + 'vtest.avi')
bs = BackgroundSegmentation(video, 6e-4, 0.2, 20, 0.1, bboxes=[(40, 140), (80, 80)], removeNoise=True)
# fig = plt.figure(figsize=(10, 7))
rows = 1
columns = 2
files = os.listdir(input_dir)
files.sort()
num_files = len(files)
# gtVid = Video(output_dir)               
IoUscore = []
# bs.verifyStdDev()
for i in tqdm(range(num_files)):
    pred, prob, bboxes= bs.runSegment()
    for bbox in bboxes:
        cv2.rectangle(pred, (bbox.x1, bbox.y2),(bbox.x2, bbox.y1), (0, 1, 0), 1 )
    cv2.imwrite(save_dir+files[i] , pred*255)
    # print(pred.dtype, groundTruth.dtype)
    # intersection = np.bitwise_and(pred, groundTruth).ravel()
    # union = np.bitwise_or(pred, groundTruth).ravel()
    # x = np.sum(union)
    # if x!= 0:
    #     IoUscore.append(np.sum(intersection)/x)
    # else:
    #     IoUscore.append(1)
# IoUscore = np.array(IoUscore)
# print(IoUscore.mean())

