import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from collections import deque

plt.rcParams['figure.figsize'] = [15, 10]

frames_to_calc = 5

# ShiTomasi corner detection. Adjusted the parameters so it would track better.
config_st = {
    'maxCorners': 200,
    'qualityLevel': 0.1,
    'minDistance': 10,
    'blockSize': 7,
    'useHarrisDetector': False
}

# Lucas-Kanade optical flow
config_lk = {
    'winSize': (15, 15),
    'maxLevel': 2,
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    'minEigThreshold': 0.001
 }

testfile = 'nfs/testfile.mp4'

video = cv2.VideoCapture(testfile)

def show_good_features(frame_source, corners):
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(frame_source, (int(x), int(y)), 5, (0, 0, 255), -1)

def show_lines(frame_source, source, destination):
    random_colors = np.random.randint(0, 255, (100, 3))
    for i, (dst, src) in enumerate(zip(source, destination)):
        x_dst, y_dst = dst
        x_src, y_src = src

        color = random_colors[i % 100].tolist()

        cv2.arrowedLine(frame_source, (int(x_src), int(y_src)), (int(x_dst), int(y_dst)), color, 2, tipLength=0.5)
        cv2.circle(frame_source, (int(x_src), int(y_src)), 10, color, -1)

prev_gray_frames = deque(maxlen=frames_to_calc)

while True:
    ret, frame = video.read()

    if not ret:
        print("End of video.")
        break

    src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p_src = cv2.goodFeaturesToTrack(src_gray, mask=None, **config_st)

    output_frame = frame.copy()

    if len(prev_gray_frames) >= frames_to_calc:
        prev_frame_gray = prev_gray_frames.popleft()
    else:
        prev_frame_gray = None

    if prev_frame_gray is not None:
        p_dst, status, err = cv2.calcOpticalFlowPyrLK(src_gray, prev_frame_gray, p_src, None, **config_lk)

        # Select points that have been successfully tracked
        if p_dst is not None:
            p_dst = p_dst[status == 1]
            p_src = p_src[status == 1]

            show_lines(output_frame, p_src, p_dst)

    show_good_features(output_frame, p_src)

    cv2.imshow('Tracking ' + testfile, output_frame)

    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

    prev_gray_frames.append(src_gray)
    time.sleep(0.1)

cv2.destroyAllWindows()
print("Tracking ended.")