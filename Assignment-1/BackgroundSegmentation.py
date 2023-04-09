import os
from scipy.ndimage import label
from KDE import *
import cv2
class Node:
    '''
    Building block of linked list data structure
    '''
    def __init__(self):
        self.next = None
        self.data = None
    def __str__(self):
        '''
        String Representation of node
        '''
        return str(self.data)
class LinkedList:   
    '''
    Implemenation of Queue data structure using Linked List Data structure
    '''
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    def add(self, data):
        '''
        Add a node at tail with value stored as data
        '''
        self.size += 1
        if self.head == None:
            self.head = Node()
            self.head.data = data
            self.tail = self.head
            return
        self.tail.next = Node()
        self.tail.next.data = data
        self.tail = self.tail.next
    def delete(self):
        '''
        Delete the node at head
        '''
        self.size -= 1
        self.head = self.head.next
        return
    def __iter__(self):
        self.trav = self.head
        return self
    def __next__(self):
        if self.trav == None:
            raise(StopIteration)
        data = self.trav.data
        self.trav = self.trav.next
        return data
    def __str__(self):
        '''
        String Representation of list
        '''
        return '[' + ','.join([str(i) for i in self]) + ']'
    def __len__(self):
        return self.size
class Video:
    '''
    Object to load frames from folder
    '''
    def __init__(self, frameFolder, ):
        self.frames = os.listdir(frameFolder)
        self.frames.sort()
        self.frameFolder = frameFolder
        self.index = 0
    def read(self):
        if self.index < len(self.frames):
            self.index += 1
            return True, cv2.imread(self.frameFolder +'/' + self.frames[self.index - 1]) #cv2.cvtColor(cv2.imread(self.frameFolder +'/' + self.frames[self.index - 1]), cv2.COLOR_BGR2GRAY)
        return False, None
    def toVideo(self, filename):
        size = cv2.imread(self.frameFolder +'/' + self.frames[0]).shape
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(filename ,fourcc, 15,(size[1], size[0]))
        for i in range(len(self.frames)):
            out.write(cv2.imread(self.frameFolder +'/' + self.frames[i]))
        out.release()

class BackgroundSegmentation:
    '''
    Background Segmentation class

    This separates background from foreground using Kernel Density Estimation(KDE)
    with exponentially decaying weights.
    This implementation is for the assignment 1 of course COL780-Computer vision
    
    Parameters:
    
    video: Video Object
    
    threshold: Probability threshold. Detect background if P > threshold
    
    alpha: The exponential decay factor
    
    num_frames: Initial number of frames
    '''
    def __init__(self, video, threshold, alpha, num_frames, bthresh  = 1e-2, removeNoise = False,bboxes = []) :
        self.video = video
        self.threshold = threshold
        self.alpha = alpha
        self.num_frames = num_frames
        self.std_dev = None
        self.prevframes = LinkedList()
        # self.bboxes  = 
        self.bboxes = bboxes
        self.bthresh = bthresh
        self.removeNoise = removeNoise
    def runSegment(self):
        '''
        Background Subtraction in the next frame.
        Predict the value
        '''
        _,frame = self.video.read()
        # frame = cv2.cvtColor()
        frame = frame.astype(np.float16)
        # frame = frame/255
        if len(self.prevframes) in [0, 1]:
            prediction = np.zeros(frame.shape)
            prediction = prediction.astype(np.uint8)
            self.prevframes.add(frame)
            return prediction, prediction, []
        
        # if self.video.index < self.num_frames// 2 or self.video.index % self.num_frames == self.num_frames//2:
        self.std_dev = np.zeros(frame.shape)
        self.std_dev = self.updateStdDev()
        # print("std_dev" , self.std_dev[10, 10,0])
        probabilities = ExpAvgProbability(frame, self.prevframes, self.std_dev, self.alpha)
        # probabilities = probabilities/len(self.prevframes)
        # print(np.min(probabilities), np.max(probabilities))
        # probabilities[probabilities>1] = 10
        # probabilities = EqWtProbability(frame, self.prevframes, self.std_dev)
        prediction = (probabilities < self.threshold).astype(np.uint32)
        if self.removeNoise:        
            integralImages = self.calculateIntegralImages(prediction)
            prediction = self.foregroundAggregation(integralImages, prediction)
            if len(self.bboxes) != 0:
                integralImages = self.calculateIntegralImages(prediction)
                bboxes = self.getBoundingBoxes(integralImages)
                bboxes = self.nonMaxSupression(bboxes)
                prediction = self.complemnet(bboxes, prediction)
            else:
                bboxes = []
        else:
            bboxes = []
        ans = np.zeros((*prediction.shape, 3))
        if len(self.prevframes) == self.num_frames:
            self.prevframes.delete()
        self.prevframes.add(frame)

        for i in range(3):
            ans[:, :, i] = prediction
        ans = ans.astype(np.uint8)
       
        return ans, probabilities, bboxes
    def getBoundingBoxes(self, integralImages):
        x, y = integralImages.shape
        ans = []
        for a,b in self.bboxes:
            for i in range(b - 1, x):
                for j in range(a - 1, y):
                    x2 = j
                    y2 = i
                    x1 = j + 1 - a 
                    y1 = i + 1 - b 
                    val1, val2, val3 = 0,0,0
                    if x1 >0:
                        val1 = integralImages[y2, x1 - 1]
                    if y1 >0:
                        val2 = integralImages[y1 - 1, x2]
                    if x1 >0 and y1 >0:
                        val3 = integralImages[y1 - 1, x1 - 1]
                    prob = integralImages[y2, x2] +val3 - val2 - val1
                    
                    if prob > self.bthresh * (a * b):
                        # print(prob/(a*b), x1, y1, x2, y2)
                        ans.append(BoundingBox(x1,y1,x2,y2, prob/(a * b)))
        # for a,b in self.bboxes:
        #     for i in range(x - b):
        #         for j in range(y - a):
        #             x2 = j + b
        #             y2 = i + a
        #             x1 = j 
        #             y1 = i 
        #             val1, val2, val3 = 0,0,0
        #             if x1 >0:
        #                 val1 = integralImages[y2, x1 - 1]
        #             if y1 >0:
        #                 val2 = integralImages[y1 - 1, x2]
        #             if x1 >0 and y1 >0:
        #                 val3 = integralImages[y1 - 1, x1 - 1]
        #             prob = integralImages[y2, x2] +val3 - val2 - val1
                    
        #             if prob > self.bthresh * (a * b):
        #                 # print(prob/(a*b), x1, y1, x2, y2)
        #                 ans.append(BoundingBox(x1,y1,x2,y2, prob/(a * b)))
        return ans
    def nonMaxSupression(self, bboxes):
        bboxes.sort(key = lambda x: x.prob, reverse = True)
        ans = []
        while len(bboxes) >0:
            ans.append(bboxes[0])
            temp = []
            for box in bboxes:
                if box.IoU(bboxes[0]) <0.05:
                    temp.append(box)
            bboxes = temp
        return ans
    def complemnet(self, bboxes, image):
        ans = np.zeros(image.shape)
        for bbox in bboxes:
            ans[ bbox.y1 : bbox.y2 + 1, bbox.x1:bbox.x2 + 1] = image[ bbox.y1 : bbox.y2 + 1, bbox.x1:bbox.x2 + 1]
        return ans.astype(np.uint8)
    def updateStdDev(self):
        '''
        Find the standard deviations of each pixel
        '''
        arr = np.zeros((len(self.prevframes) - 1, *self.std_dev.shape))
        prev = None
        for idx,frame in enumerate(self.prevframes):
            if idx !=0:
                arr[idx - 1,] = np.abs(frame - prev)
            prev = frame
        self.std_dev = np.median(arr, axis = 0)
        self.std_dev = self.std_dev /(0.68 * np.sqrt(2))
        # arr = arr.astype(np.float32)
        return  self.std_dev
    def calculateIntegralImages(self, probabilities):
        dp = np.zeros(probabilities.shape)
        x, y = probabilities.shape
        for i in range(x):
            for j in range(y):
                dp[i][j] = probabilities[i][j]
                if i >0:
                    dp[i][j] += dp[i - 1][j]
                if j > 0:
                    dp[i][j] += dp[i][j-1]
                if i>0 and j>0:
                    dp[i][j] -= dp[i -1][j -1]
        return dp
    def foregroundAggregation(self, integralImages, prediction):
        # y, x = prediction.shape
        # ans = np.zeros((y, x))
        # ans[:, :] = prediction
        # for i in range(9, y):
        #     for j in range(9, x):
        #         if prediction[i-5, j-5] == 1.0:
        #             x2 = j
        #             y2 = i
        #             x1 = j + 1 - 10 
        #             y1 = i + 1 - 10 
        #             val1, val2, val3 = 0,0,0
        #             if x1 >0:
        #                 val1 = integralImages[y2, x1 - 1]
        #             if y1 >0:
        #                 val2 = integralImages[y1 - 1, x2]
        #             if x1 >0 and y1 >0:
        #                 val3 = integralImages[y1 - 1, x1 - 1]
        #             prob = integralImages[y2, x2] +val3 - val2 - val1
        #             if prob < 20:
        #                 ans[i-5,j-5] = 0
        # return ans
        # prediction = np.ones(prediction.shape)
        ans = np.zeros(prediction.shape)
        y, x = prediction.shape
        ans[:, :] = prediction
        for i in range(9, y):
            for j in range(9, x):
                if prediction[i - 4, j - 4] == 1:
                    sm = np.sum(prediction[i - 9:i + 1, j- 9: j + 1].ravel())
                    if sm < 20:
                        ans[i-4, j-4] = 0
                # ans[i - 4, j- 4] = 1
        return ans


    def verifyStdDev(self):
        shape = None
        for _ in range(60):
            _,frame = self.video.read()
            shape = frame.shape
             # frame = cv2.cvtColor()
            frame = frame.astype(np.float128)
            self.prevframes.add(frame)
        a,b, _ = shape
        self.std_dev = np.zeros(shape)
        a = np.random.randint(0,a )
        b = np.random.randint(0, b)
        import matplotlib.pyplot as plt
        li = np.array([frame[a, b, 0] for frame in self.prevframes])
        self.std_dev = self.updateStdDev()
        std_dev = self.std_dev[a, b, 0]
        mean = li.mean()
        plt.hist(li, bins = range(256), density = True)
        x = np.array(list(range(256)))
        def ker(x, mean, sigma):
            return 1/(np.sqrt(2*np.pi * sigma)) * np.exp( -0.5 *(( x- mean )**2)/(sigma ** 2))
        plt.plot(x, ker(x, mean, std_dev))
        plt.show()



        
    # def updateStdDev(self):
        # arr = np.zeros((len(self.prevframes)//2 , *self.std_dev.shape))
        # prev = None
        # for idx,frame in enumerate(self.prevframes):
        #     if idx %2 ==0:
        #         prev = frame
        #     else:
        #         arr[idx//2] = np.abs(frame - prev)+ 1e-9
        #     prev = frame
        # self.std_dev = np.median(arr, axis = 0)
        # self.std_dev = self.std_dev /(0.68 * np.sqrt(2))
        # # arr = arr.astype(np.float32)
        # return  self.std_dev
        
class BoundingBox:
    '''
    Bounding Box class
    (x1, y1) are the coordinates of the top left corner
    (x2, y2) are the coordinates of the bottom right corner
    '''
    def __init__(self, x1, y1, x2, y2 , prob = 0.0):
        self.x1 = x1 
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.prob = prob
    def __str__(self):
        return '[' + ','.join([str(self.x1), str(self.y1), str(self.x2), str(self.y2)]) + ']'
    def area(self):
        return (self.x2 - self.x1 + 1) * (self.y2 - self.y1 + 1)
    def IoU(self, box2):
        x11 = max(self.x1, box2.x1)
        y11 = max(self.y1, box2.y1)
        x22 = min(self.x2, box2.x2)
        y22 = min(self.y2, box2.y2)
        if x22< x11 or y22< y11:
            return 0
        intersection_area = (x22 - x11 + 1) * (y22 - y11 + 1)
        union_area = self.area() +  box2.area() - intersection_area
        return intersection_area/union_area
    
