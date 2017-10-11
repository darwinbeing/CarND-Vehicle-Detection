import os
import cv2
from moviepy.editor import VideoFileClip
from utils import *
from pipeline import *

VIDEO_OUTPUT_DIR = 'test_videos_output/'

def process_image(image):
    result = pipeline(image)
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(result)
    # plt.show()

    return result

def process_video(video_input, video_output):
    clip = VideoFileClip(os.path.join(os.getcwd(), video_input))
    processed = clip.fl_image(process_image)
    processed.write_videofile(os.path.join(VIDEO_OUTPUT_DIR, video_output), audio=False)


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

images = glob.glob('test_images/*.jpg')
# images = glob.glob('test_images/test5.jpg')
for idx, fname in enumerate(images):
    image = cv2.imread(fname)
    image = BGR2RGB(image)

    vehicleTracker.threshold = 2
    result = pipeline(image)

    vehicleTracker.threshold = 5
    vehicleTracker.heatmaps.clear()

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(result)

    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


if not os.path.isdir(VIDEO_OUTPUT_DIR):
    os.mkdir(VIDEO_OUTPUT_DIR)

process_video('project_video.mp4', 'project_video.mp4')
process_video('test_video.mp4', 'test_video.mp4')
