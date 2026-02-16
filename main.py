"""
File: main.py
Author: Victor Manuel Vazquez Morales

Description: This simple algorithm computes a series of steps to allow the
segmentation of cars on a specific video, implementing a traffic-flow monitoring.
"""

import numpy as np
import cv2
import os


def main(argv=None) -> None:

    ## Enhancing input video. Reduces noise using blurring
    file = "RoadTraffic.mp4"
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        print(f"Error: Couln't open file video [{file}]")
        exit()

    enhanced_file = "enhanced.avi"
    if not os.path.isfile(enhanced_file):
        print("Enhancing original video")
        enhance_video(cap, enhanced_file)
    else:
        print(f"Reading {enhanced_file}")

    cap.release()

    # Background extraction. Computes background
    cap = cv2.VideoCapture(enhanced_file)
    if not cap.isOpened():
        print(f"Error: Couldn't open file video [{enhanced_file}]")
        exit()

    background_file = "background.jpeg"
    if not os.path.isfile(background_file):
        print("Extracting background")
        compute_background(cap, background_file)
    else:
        print(f"Reading {background_file}")

    # Car segmentation. Segments image and generates output video
    bg = cv2.imread(background_file)
    output_video = "output.avi"
    if not os.path.isfile(output_video):
        print("Generating output video...")
        traffic_flow(cap, bg, output_video)
    else:
        print(f"{output_video} already exists!")

    cap.release()
    cv2.destroyAllWindows()


def enhance_video(v: cv2.VideoCapture, output_file: str) -> None:
    """
    This function applies a blurring process to enhance the input video, which
    actually has noise.

    Args:
        v (cv2.VideoCapture): Video capture object of the input video.
        output_file (str): Name of the output file.
    """

    frame_width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(v.get(cv2.CAP_PROP_FPS))

    # Define the codec, create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = v.read()
        if not ret:
            break
        blurred = cv2.blur(frame, (5, 5))
        out.write(blurred)

    v.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Set current frame as 0


def compute_background(v: cv2.VideoCapture, output_img: str) -> None:
    """
    Given an input video, this function extracts the background calculating the average
    for all the frames and saves the result in an image named as the second argument.

    Args:
        v (cv2.VideoCapture): Video used to extract the background.
        output_img (str): Output image file name.
    """

    ret, background = v.read()
    if not ret:
        return

    # Convert data type to a float64 matrix, necessary for backgrond computation
    background = np.float64(background)

    while True:
        ret, frame = v.read()
        if not ret:
            break
        background += frame
    # Compute the background using the average of all frames
    background = background / float(v.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.imwrite(output_img, np.uint8(background))

    v.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Set current frame as 0


def segment_image(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    """
    Given an input frame, this method isolated the moving objects (cars) by substracting
    the background image. It also applies binarization and a closing morphological operation
    to generate the output mask.

    Args:
        frame (np.ndarray)
        background (np.ndarray)

    Returns:
        np.ndarray : Output mask (binary image)
    """

    foreground = cv2.absdiff(frame, background)
    img_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(img_gray, 35, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Draw strategically a line to dive both lanes, avoding overlapping
    cv2.line(mask, (0, 250), (1155, 100), 0, 3)
    return mask


def traffic_flow(v: cv2.VideoCapture, background: np.ndarray, output_file: str) -> None:
    """
    Docstring for traffic_flow

    :param v: Description
    :type v: cv2.VideoCapture
    :param background: Description
    :type background: np.ndarray
    :param output_file: Description
    :type output_file: str
    """

    frame_width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(v.get(cv2.CAP_PROP_FPS))

    # Define the codec, create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = v.read()
        if not ret:
            break
        mask = segment_image(frame, background)
        contours, _ = cv2.findContours(mask, 1, 2)
        flow_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 6000:
                x, y, w, h = cv2.boundingRect(cnt)
                # Draw bounding boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 125), 2)
                flow_count += 1
        # Write the amount of cars detected in current frame
        cv2.rectangle(frame, (0, 0), (160, 50), 0, -1)
        cv2.putText(
            frame,
            f"Cars: {flow_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.imshow("output", frame)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

        out.write(frame)
    v.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Set current frame as 0


main()
