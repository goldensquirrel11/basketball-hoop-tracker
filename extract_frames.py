import cv2
from pathlib import Path
import argparse

"""
    Extracts frames from a video and saves them as JPEG images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Path to the directory where frames will be saved.
"""
def extract_frames(video_path:Path, output_dir:Path):
    
    # Create output directory if it does not exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    video_feed = cv2.VideoCapture(video_path)

    total_frames = int(video_feed.get(cv2.CAP_PROP_FRAME_COUNT))

    ret,frame = video_feed.read()
    while(ret):    
        currentframe = int(video_feed.get(cv2.CAP_PROP_POS_FRAMES))
        print (f'Extracting frame {currentframe}/{total_frames} from video: ' + str(video_path.name))

        name = output_dir / f'{video_path.stem}_{currentframe:06d}.jpg'

        # write extracted image to output directory
        cv2.imwrite(name, frame)

        ret,frame = video_feed.read()
    
    video_feed.release()

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument('video_path', type=Path, help='Path to the input video file.')
    parser.add_argument('output_dir', type=Path, help='Directory to save the extracted frames.')
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_dir)