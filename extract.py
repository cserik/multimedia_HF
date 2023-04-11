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

    interval = max(1, frame_count // 100)

    frames = []
    for i in range(0, 100):
        frame_index = i * interval
        if frame_index >= frame_count:
            # If the video has fewer than 100 frames, loop back to the beginning
            frame_index %= frame_count

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and convert to grayscale
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames.append(frame)

    cap.release()

    # If the video has fewer than 100 frames, repeat the last frame until there are 100 frames
    while len(frames) < 100:
        frames.append(frames[-1])

    for i, frame in enumerate(frames):
        image_name = f"{video_name}_{i:05d}.jpg"
        image_path = os.path.join(image_dir, image_name)
        cv2.imwrite(image_path, frame)

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
