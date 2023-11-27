import os
import cv2

image_folder = './logs/elastic_pendulum_encoder-decoder-64_1/predictions'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.startswith("4_")]
image_nums = [(int(img[2:-4]), img) for img in images]
image_nums.sort()
images = [img[1] for img in image_nums]
print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()