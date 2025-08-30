import cv2
import numpy as np

cap = cv2.VideoCapture("contornos/camara/video_1_12.avi")

total_figures = 0
circles = 0
rings = 0
figure_present = False
start_frame = 0
frames_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_figure = len(contours) > 0

    if has_figure and not figure_present:
        figure_present = True
        start_frame = len(frames_buffer)

    elif not has_figure and figure_present:
        figure_present = False
        end_frame = len(frames_buffer)
        mid_frame_idx = (start_frame + end_frame) // 2

        if mid_frame_idx < len(frames_buffer):
            classified = False
            frame_offset = 0

            while not classified and frame_offset < len(frames_buffer):
                test_idx = (
                    mid_frame_idx - frame_offset
                    if frame_offset % 2 == 0
                    else mid_frame_idx + (frame_offset + 1) // 2
                )

                if 0 <= test_idx < len(frames_buffer):
                    test_frame = frames_buffer[test_idx]
                    test_gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
                    _, test_binary = cv2.threshold(
                        test_gray, 50, 255, cv2.THRESH_BINARY
                    )

                    contours, hierarchy = cv2.findContours(
                        test_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if hierarchy is not None:
                        internal_contours = 0
                        for i in range(len(hierarchy[0])):
                            if hierarchy[0][i][3] != -1:
                                internal_contours += 1

                        if internal_contours == 1 or internal_contours == 2:
                            total_figures += 1
                            if internal_contours == 1:
                                circles += 1
                                detection_img = test_binary.copy()
                                cv2.putText(
                                    detection_img,
                                    f"Simple #{circles}",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2,
                                )
                            elif internal_contours == 2:
                                rings += 1
                                detection_img = test_binary.copy()
                                cv2.putText(
                                    detection_img,
                                    f"Doble #{rings}",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2,
                                )

                            cv2.imshow(f"Detection {total_figures}", detection_img)
                            cv2.waitKey(500)
                            classified = True
                        elif internal_contours > 2:
                            for back_idx in range(end_frame - 10, end_frame):
                                if back_idx >= 0 and back_idx < len(frames_buffer):
                                    back_frame = frames_buffer[back_idx]
                                    back_gray = cv2.cvtColor(
                                        back_frame, cv2.COLOR_BGR2GRAY
                                    )
                                    _, back_binary = cv2.threshold(
                                        back_gray, 50, 255, cv2.THRESH_BINARY
                                    )

                                    back_contours, back_hierarchy = cv2.findContours(
                                        back_binary,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE,
                                    )

                                    if back_hierarchy is not None:
                                        back_internal = 0
                                        for i in range(len(back_hierarchy[0])):
                                            if back_hierarchy[0][i][3] != -1:
                                                back_internal += 1

                                        if back_internal == 2:
                                            total_figures += 1
                                            rings += 1
                                            detection_img = back_binary.copy()
                                            cv2.putText(
                                                detection_img,
                                                f"Doble #{rings}",
                                                (10, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (255, 255, 255),
                                                2,
                                            )
                                            cv2.imshow(
                                                f"Detection {total_figures}",
                                                detection_img,
                                            )
                                            cv2.waitKey(500)
                                            classified = True
                                            break

                frame_offset += 1

        frames_buffer = []

    frames_buffer.append(frame.copy())

    cv2.imshow("Frame", frame)
    cv2.imshow("Binary", binary)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total: {total_figures}")
print(f"Circulos SImples: {circles}")
print(f"Circulos Dobles: {rings}")
