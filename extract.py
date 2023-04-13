import os
import cv2
import argparse
import multiprocessing as mp
import time

def process_video(video_path, image_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    image_dir = os.path.join(image_path, os.path.dirname(video_path))
    os.makedirs(image_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract 30 frames from the middle of the video
    middle_frame = frame_count // 2
    interval = max(1, (middle_frame - 15) // 30)
    frames = []
    for i in range(0, 30):
        frame_index = middle_frame - 15 + i * interval
        if frame_index >= frame_count:
            # If the video has fewer than 30 frames from the middle, loop back to the beginning
            frame_index %= frame_count

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and convert to RGB
        frame = cv2.resize(frame, (224, 224))

        # Save the frame
        image_name = f"{video_name}_{i:05d}.jpg"
        image_path = os.path.join(image_dir, image_name)
        cv2.imwrite(image_path, frame)

        # Crop and save the face(s) in the frame using the provided function
        crop_face(image_path)

    cap.release()

def crop_face(name):
    # Read the input image
    img = cv2.imread(name)
    # read the haarcascade to detect the faces in an image
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # detects faces in the input image
    faces = face_cascade.detectMultiScale(img, 1.3, 4)
    #print('Number of detected faces:', len(faces))

    # loop over all detected faces
    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y + h, x:x + w]
            cv2.imwrite(name, face)

def extract_frames(video_dir, image_dir, num_workers=4):
    video_paths = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith(".avi"):
                video_path = os.path.join(root, file)
                video_paths.append(video_path)

    with mp.Pool(num_workers) as pool:
        results = [pool.apply_async(process_video, args=(video_path, image_dir)) for video_path in video_paths]

        while True:
            remaining = sum([1 for result in results if not result.ready()])
            if remaining == 0:
                break

            for _ in range(remaining):
                print(".", end="", flush=True)
                time.sleep(0.5)

    print("\nDone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_dir", help="Path to the video directory")
    parser.add_argument("-i", "--image_dir", help="Path to the image directory")
    parser.add_argument("-w", "--num_workers", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    extract_frames(args.video_dir, args.image_dir, args.num_workers)
