import cv2
import numpy as np
import time
from ultralytics import YOLO
import util as ut

cap = cv2.VideoCapture(0)

threshold = 0.5
y_tolerance = 0.04
x_tolerance = 0.1
red_ratio_threshold = 4

x_deviation = 0
y_center = 0
object_to_track = 0

Kp = 180
base_speed = 150

left_speed_out = 0
right_speed_out = 0

# SEARCHING
last_seen_time = time.time()
last_rotate_dir = 1      # 1: quay phải, -1: quay trái
lost_timeout = 10.0       # mất dấu dưới 2s thì quay tìm
search_speed = 80


def move_robot():
    global x_deviation, y_center, y_tolerance, x_tolerance
    global Kp, base_speed
    global left_speed_out, right_speed_out
    global last_seen_time, last_rotate_dir, lost_timeout, search_speed

    error = 0 if abs(x_deviation) <= x_tolerance else x_deviation
    pid_value = Kp * error

    if y_center == 0:
        # Mất dấu thì quay tìm lại trong lost_timeout giây
        if time.time() - last_seen_time < lost_timeout:

            if last_rotate_dir > 0:
                # quay phải
                left_speed = search_speed
                right_speed = 0
            else:
                # quay trái
                left_speed = 0
                right_speed = search_speed
        else:
            left_speed = 0
            right_speed = 0

    else:
        if y_center > (0.5 + y_tolerance):
            current_base = base_speed

        elif (0.5 - y_tolerance) <= y_center <= (0.5 + y_tolerance):
            current_base = 0

        else:
            current_base = 0

        # Quay tại chỗ khi người lệch khỏi vùng X tolerance
        if current_base == 0 and abs(error) > x_tolerance:


            if pid_value > 0:
                # người ở bên trái ảnh -> quay trái
                left_speed = 0
                right_speed = int(55 + abs(pid_value))
                last_rotate_dir = -1
            else:
                # người ở bên phải ảnh -> quay phải
                left_speed = int(55 + abs(pid_value))
                right_speed = 0
                last_rotate_dir = 1

        else:
            left_speed = int(current_base - pid_value)
            right_speed = int(current_base + pid_value)

            if error > 0:
                last_rotate_dir = -1
            elif error < 0:
                last_rotate_dir = 1

    left_speed = max(0, min(255, left_speed))
    right_speed = max(0, min(255, right_speed))

    left_speed_out = left_speed
    right_speed_out = right_speed

    try:
        ut.set_speeds(left_speed, right_speed)
    except AttributeError:
        print(f"Lệnh ảo -> L:{left_speed} | R:{right_speed} ")


def track_object(results, frame, mask):
    global x_deviation, y_center, y_tolerance, x_tolerance
    global last_seen_time, last_rotate_dir

    h, w, c = frame.shape

    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 0, 0), 2)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 2)

    left_line_x = int(w * (0.5 - x_tolerance))
    right_line_x = int(w * (0.5 + x_tolerance))
    cv2.line(frame, (left_line_x, 0), (left_line_x, h), (0, 255, 255), 1)
    cv2.line(frame, (right_line_x, 0), (right_line_x, h), (0, 255, 255), 1)

    top_line_y = int(h * (0.5 - y_tolerance))
    bottom_line_y = int(h * (0.5 + y_tolerance))
    cv2.line(frame, (0, top_line_y), (w, top_line_y), (0, 255, 255), 1)
    cv2.line(frame, (0, bottom_line_y), (w, bottom_line_y), (0, 255, 255), 1)

    cv2.putText(frame, "Stop Zone", (10, top_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    boxes = results[0].boxes

    if len(boxes) == 0:
        y_center = 0
        x_deviation = 0
        move_robot()
        return

    flag = 0
    selected = False

    for box in boxes:
        if int(box.cls[0]) == object_to_track:
            x_min, y_min, x_max, y_max_obj = box.xyxyn[0].tolist()

            x1 = int(x_min * w)
            y1 = int(y_min * h)
            x2 = int(x_max * w)
            y2 = int(y_max_obj * h)

            check_y2 = y1 + int((y2 - y1) * 4 / 5)
            # Vẽ vạch 3/4 thân trên để k iểm tra màu đỏ
            cv2.line(frame, (x1, check_y2), (x2, check_y2), (255, 0, 255), 2)
            cv2.putText(frame, "3/4 check line", (x1, check_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            p_mask = mask[max(0, y1):min(h, check_y2),
                          max(0, x1):min(w, x2)]

            ratio = (
                cv2.countNonZero(p_mask) / p_mask.size * 100
            ) if p_mask.size > 0 else 0

            color = (0, 255, 0) if ratio > red_ratio_threshold else (0, 0, 255)
            label = f"Check: {'OK' if ratio > red_ratio_threshold else 'FAIL'} ({ratio:.1f}%)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if ratio <= red_ratio_threshold:
                continue

            if selected:
                continue

            selected = True
            flag = 1

            x_diff = x_max - x_min
            y_diff = y_max_obj - y_min

            obj_x_center = x_min + (x_diff / 2)
            obj_y_center = y_min + (y_diff / 2)

            center_x_px = int(obj_x_center * w)
            center_y_px = int(obj_y_center * h)

            cv2.circle(frame, (center_x_px, center_y_px), 5, (255, 255, 0), -1)

            obj_x_center = round(obj_x_center, 3)
            obj_y_center = round(obj_y_center, 3)

            x_deviation = round(0.5 - obj_x_center, 3)
            y_center = obj_y_center

            last_seen_time = time.time()

            if x_deviation > x_tolerance:
                last_rotate_dir = -1
            elif x_deviation < -x_tolerance:
                last_rotate_dir = 1

    if flag == 0:
        y_center = 0
        x_deviation = 0

    move_robot()


def main():

    try:
        print("Đang tải model YOLOv8...")
        model = YOLO("yolov8n.pt")
        print("Tải model thành công!")
    except Exception as e:
        print(f"LỖI TẢI MODEL: {e}")
        cap.release()
        return

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Không thể kết nối với Camera!")
            break

        cv2_im = frame

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask = cv2.add(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        results = model.track(
            cv2_im,
            conf=threshold,
            classes=[0],
            verbose=False,
            persist=True
        )

        track_object(results, cv2_im, mask)

        process_time = time.time() - start_time
        fps = round(1.0 / process_time, 1) if process_time > 0 else 0

        cv2.putText(cv2_im, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        cv2.putText(cv2_im, f"L_SPEED: {left_speed_out}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.putText(cv2_im, f"R_SPEED: {right_speed_out}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.putText(cv2_im, f"X: {x_deviation} | Y: {y_center}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("CAMERA", cv2_im)
        cv2.imshow("Red Mask Filter", mask)

        if cv2.waitKey(1) & 0xFF == 27:
            try:
                ut.set_speeds(0, 0)
            except:
                pass
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()