import cv2
from matplotlib import pyplot as plt
import time

plt.rcParams['figure.figsize'] = [15, 10]

def draw_rect(frame_2_draw, rect, color, text, index):
    x, y, w, h = [int(v) for v in rect]
    cv2.rectangle(frame_2_draw, (x, y), (x + w, y + h), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    dx = [0, 1, 0, 1]
    dy = [0, 0, 1, 1]

    cv2.putText(frame_2_draw, text,
    (x + (dx[index % 4]) * w, y + (dy[index % 4]) * h),
        font, font_scale, color, font_thickness
    )


def overlay_image(background, foreground, x, y):
    fg_height, fg_width = foreground.shape[:2]
    bg_height, bg_width = background.shape[:2]
    w = min(fg_width, bg_width - x)
    h = min(fg_height, bg_height - y)
    if w < fg_width or h < fg_height:
        foreground = foreground[:h, :w]
    background[y:y + h, x:x + w] = foreground[:h, :w]

    return background

testfile = 'nfs/testfile.mp4'
output_file = 'nfs/output.mp4'
video = cv2.VideoCapture(testfile)

ret, frame = video.read()

if not ret:
    raise ValueError("Could not read video")

roi = cv2.selectROI("Tracking", frame, False)
cv2.destroyWindow("Tracking")

roi_image = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])].copy()

trackers = [
    {
        'name': 'KCF',
        'tracker': cv2.TrackerKCF_create(),
        'color': (0, 0, 255)
    },
    {
        'name': 'CSRT',
        'tracker': cv2.TrackerCSRT_create(),
        'color': (255, 0, 0)
    },
    {
        'name': 'MIL',
        'tracker': cv2.TrackerMIL_create(),
        'color': (255, 255, 0)
    }
]

for tracker in trackers:
    tracker['tracker'].init(frame, roi)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
fg_height, fg_width = frame.shape[:2]
video_writer = cv2.VideoWriter(output_file, fourcc, 30.0, (fg_width, fg_height))

while True:
    ret, frame = video.read()

    if not ret:
        print("End of video.")
        break

    bboxes = []

    for tracker in trackers:
        traker = tracker['tracker']
        success, bbox = traker.update(frame)

        if success:
            bboxes.append({
                'tracker': tracker['name'],
                'bbox': bbox,
                'color': tracker['color']
            })


    output_frame = frame.copy()
    overlay_image(output_frame, roi_image, 0, 0)

    for i, bbox in enumerate(bboxes):
        draw_rect(output_frame, bbox['bbox'], bbox['color'], bbox['tracker'], i)

    cv2.imshow('Tracking ' + testfile, output_frame)

    video_writer.write(output_frame)

    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

    time.sleep(0.1)

cv2.destroyAllWindows()
video_writer.release()

print("Tracking ended.")