# (Necessary) Imports
import pandas as pd
import os
import glob
import time

import cv2



# Utility based functions
def calculate_elapsed_time(start_time):
    """Calculating elapsed time in HH:MM:SS format."""

    elapsed_time = int(time.time() - start_time)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{str(hours).zfill(2)}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}"

def select_tracker_type():
    """Prompting the user to select a tracker type."""

    # Available single object trackers types; for single object tracking
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']

    print("\n\nAvailable tracking algorithms:")
    for i, tracker_type in enumerate(tracker_types):
        print(f"{i + 1}: {tracker_type}")
    
    try:
        tracker_choice = int(input("\nEnter the number corresponding to the tracking algorithm you want to use: "))
        if not (1 <= tracker_choice <= len(tracker_types)):
            raise ValueError("Invalid tracker choice")
    
        return tracker_types[tracker_choice - 1]
    except (ValueError, IndexError) as e:
        print(f"Error: {e}. Please enter a valid number between 1 and {len(tracker_types)}.")
    
        return select_tracker_type()

def select_video_file():
    """Prompting the user to select a video file format."""

    # cv2 supported video file formats
    video_file_formats = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]

    video_file_name = input("Please, input the name (path) of the video file you want to track objects from: ")
    
    print("\n\nSupported file formats:")
    for i, file_type in enumerate(video_file_formats):
        print(f"{i + 1}: {file_type}")
    
    try:
        video_file_formats_choice = int(input("\nEnter the number corresponding to the supported file format you want to use: "))
        if not (1 <= video_file_formats_choice <= len(video_file_formats)):
            raise ValueError("Invalid file format choice")
    
        return f"{video_file_name}{video_file_formats[video_file_formats_choice - 1]}"
    except (ValueError, IndexError) as e:
        print(f"Error: {e}. Please enter a valid number between 1 and {len(video_file_formats)}.")
    
        return select_video_file()

def initialize_tracker(tracker_type):
    """Initializing the tracker based on user selection."""
    
    try:
        if tracker_type == 'BOOSTING':
            return cv2.legacy.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            return cv2.legacy.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            return cv2.legacy.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            return cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            return cv2.legacy.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()
        else:
            raise ValueError("Unsupported tracker type selected.")
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        
        return None

# Implementation based function
def process_video(tracker, video_path):
    """Processing the video for object tracking."""

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("\n\nCould not process the video file")
    
        return

    # Opening the video file
    cap = cv2.VideoCapture(video_path)

    # Exiting if the video is not opened
    if not cap.isOpened():
        print("\n\nCould not process the video file")

        exit()

    # Variables to manage the selection of the bounding box
    bbox_selected = False
    bbox = None

    # Calculating the total number of frames present within the video file
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Creating a seperate window to display the frames; correctly
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

    count=0

    print()
    print()

    start_time = time.time()

    prev_bbox = None

    while True:
        time_str=calculate_elapsed_time(start_time)

        print(f"Processing frame: {count}/{total_frames}")

        count+=1    

        # Reading a frame
        ret, frame = cap.read()

        if not ret:
            print("\n\nProcess complete. Exiting...")
            
            break

        # Displaying the elapsed time for video sequence    
        cv2.putText(frame, f"Elapsed Time: {time_str}", (100, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Displaying the frame
        cv2.imshow("Frame", frame)

        # Checking if bounding box is selected
        if bbox_selected:
            # Starting the timer
            timer = cv2.getTickCount()

            # Updating tracker
            ok, bbox = tracker.update(frame)

            # Calculating frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Drawing bounding box
            if ok:
                # # Tracking success
                # p1 = (int(bbox[0]), int(bbox[1]))
                # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                # cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
                
                if prev_bbox is not None:
                    # Calculating size change ratio
                    prev_w, prev_h = prev_bbox[2], prev_bbox[3]
                    new_w, new_h = bbox[2], bbox[3]
                    
                    # Resizing threshold
                    size_change_threshold = 0.5  # Sample threshold

                    if abs(new_w - prev_w) / prev_w > size_change_threshold or abs(new_h - prev_h) / prev_h > size_change_threshold:
                        # Resizing logic here (example: increase size by 10%)
                        bbox = (bbox[0], bbox[1], new_w * 1.1, new_h * 1.1)

                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
                
                # Updating previous bbox
                prev_bbox = bbox

                # (x, y, w, h) = [int(v) for v in bbox]

                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                # Tracking failure
                cv2.putText(frame, "Tracking Process Disrupting...", (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # Calculating the elapsed time for object tracking    
            time_str1=calculate_elapsed_time(start_time1)

            # Displaying the tracker type (selected) on the frame
            cv2.putText(frame, f"Tracker: {tracker_type}", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Displaying the real-time number of frames processes and to be processed
            cv2.putText(frame, f"Processing Frame: {count} / {total_frames}", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Displaying the real-time FPS (upon which the video is being processed) on the frame
            cv2.putText(frame, f"FPS: {int(fps)}", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Displaying the elapsed time for both the video processing and object tracking sequences
            cv2.putText(frame, f"Tracking Time: {time_str1}", (100, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.putText(frame, f"Elapsed Time: {time_str}", (100, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Displaying resulting frame
            cv2.imshow("Tracking", frame)

        # Waiting for a key press to be pressed
        key = cv2.waitKey(1) & 0xFF

        # If 'p' is pressed, pausing the video to allow the user to select a bounding box; for the object to be tracked
        if key == ord('p'):
            # Displaying the roi based instructions on the paused screen
            cv2.putText(frame, "ROI Instructions: ", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.putText(frame, "- Select a ROI and then press SPACE or ENTER button!", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 20), 2)
            cv2.putText(frame, "- Cancel the selection process by pressing c button!", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 20), 2)

            bbox = cv2.selectROI("Frame", frame, False)
            tracker.init(frame, bbox)
            bbox_selected = True
            start_time1 = time.time()

        # Exiting if ESC pressed
        if key == ord('q') or key == 27:
            print("\n\nProcess interuptted...")
            
            break

    # Releasing the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()



# Driver function
if __name__ == "__main__":

    try:
        video_path = select_video_file()
        tracker_type = select_tracker_type()
        tracker = initialize_tracker(tracker_type)

        if tracker:
            process_video(tracker, video_path)
        else:
            print("Failed to initialize tracker. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
