import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.signal import find_peaks, butter, filtfilt

import os

from gaze_tracking import GazeTracking

if tf.config.list_physical_devices('GPU'):
    print("GPU is available, MediaPipe will use GPU for processing.")
else:
    print("GPU is not available, MediaPipe will use CPU for processing.")

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def seconds_to_minutes_formatter(x, pos):
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f'{minutes:02d}:{seconds:02d}'


def analyze_breathing(video_path, output_dir, time_window=2, fn=print):
    fn("###\nИнициализация определения частоты дыхания и изменения взгляда\n###")

    gaze = GazeTracking()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5)
    # Initialize cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Predefined open skin area inside the forehead
    x1 = 0.4
    x2 = 0.6
    y1 = 0.1
    y2 = 0.275

    def get_face_roi(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        if len(faces) > 0:
            return [
                faces[0][0] + int(x1 * faces[0][2]),
                faces[0][1] + int(y1 * faces[0][3]),
                faces[0][0] + int(x2 * faces[0][2]),
                faces[0][1] + int(y2 * faces[0][3])
            ]
        else:
            return [0, 0, 0, 0]

    def find_corresponding_points(landmarks, initial_bbox, image_width, image_height):
        corresponding_points = []
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            if initial_bbox[0] <= x <= initial_bbox[2] and initial_bbox[1] <= y <= initial_bbox[3]:
                corresponding_points.append(idx)
        return corresponding_points

    def update_forehead_roi(landmarks, corresponding_points, image_width, image_height):
        x_coords = []
        y_coords = []

        for idx in corresponding_points:
            x = int(landmarks[idx].x * image_width)
            y = int(landmarks[idx].y * image_height)
            x_coords.append(x)
            y_coords.append(y)

        if not x_coords or not y_coords:
            return [0, 0, 0, 0]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        return [min_x, min_y, max_x, max_y]

    def get_color_average(frame, color_id):
        if frame.size == 0:
            return 0
        return frame[:, :, color_id].sum() * 1.0 / (frame.shape[0] * frame.shape[1])

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        fn(f'Ошибка! Не удалось открыть видеофайл: {video_path}')
        return

    # Data storing
    idf = 0
    gsums = []
    timestamps = []
    gaze_x = []
    gaze_y = []
    initial_bbox = None
    corresponding_points = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_int = int(fps)
    if fps <= 0:
        fn('Ошибка! Полученный из видео FPS равен или меньше 0')
        return

    while True: # main loop
        ret, frame = cap.read()
        if not ret:
            break

        # Cascade Classifier, until face is found
        if initial_bbox is None:
            initial_bbox = get_face_roi(frame)
            if initial_bbox == [0, 0, 0, 0]:
                continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # mediapipe requires RGB image, OpenCV works with BGR
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0].landmark
            image_height, image_width, _ = frame.shape

            if not corresponding_points:
                # begin with finding mediapipe landmarks inside the POI region
                corresponding_points = find_corresponding_points(face_landmarks, initial_bbox, image_width, image_height)

            # updating POI region position using mediapipe tracking
            bbox = update_forehead_roi(face_landmarks, corresponding_points, image_width, image_height)

            if bbox != [0, 0, 0, 0]:
                roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                # highlighting roi on image
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

                green = get_color_average(roi, 1)  # 2nd channel for Green color
                gsums.append(green)
                timestamps.append(idf / fps)  # time in seconds

                # Collecting current gaze direction
                gaze.refresh(frame)
                px, py = gaze.horizontal_ratio(), gaze.vertical_ratio()
                gaze_x.append(px if px else (gaze_x[-1] if len(gaze_x) and gaze_x[-1] else 0.5))
                gaze_y.append(py if py else (gaze_y[-1] if len(gaze_y) and gaze_y[-1] else 0.5))

        idf += 1
        if idf % fps_int == 0:  # every second
            fn(f"Обработано {idf / fps:.0f} сек.")

    # Freeing resources
    cv2.destroyAllWindows()
    cap.release()


    if gsums and timestamps and len(gsums) == len(timestamps):
        # Breathing cycles (green_channel_amplitude)
        plt.figure()
        plt.plot(timestamps, gsums, 'g', label='Green Channel Amplitude')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Green Channel Amplitude')
        plt.title('Green Channel Amplitude Over Time')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, '_циклы_дыхания.png'))
        #plt.show()

        # Breathing cycles -> rate
        filtered_gsums = butter_lowpass_filter(gsums, cutoff=0.5, fs=fps)
        peaks, _ = find_peaks(filtered_gsums,
                              distance=fps * time_window)  # not less than time_window seconds between major peaks
        peak_times = np.array(timestamps)[peaks]

        if len(peak_times) > 1:
            periods = np.diff(peak_times)
            breathing_rates = 60 / periods
            rate_times = peak_times[:-1]

            if rate_times[0] > 0:
                rate_times = np.insert(rate_times, 0, 0)
                breathing_rates = np.insert(breathing_rates, 0, breathing_rates[0])
            if rate_times[-1] < timestamps[-1]:
                rate_times = np.append(rate_times, timestamps[-1])
                breathing_rates = np.append(breathing_rates, breathing_rates[-1])

            min_index = np.argmin(breathing_rates)
            max_index = np.argmax(breathing_rates)
            min_time = rate_times[min_index]
            max_time = rate_times[max_index]
            min_rate = breathing_rates[min_index]
            max_rate = breathing_rates[max_index]

            plt.figure(figsize=(15, 6), dpi=150)
            num_ticks = min(int(len(timestamps) / fps), 30)
            plt.xticks(np.linspace(rate_times[0], rate_times[-1], num_ticks), rotation=45)
            plt.gca().xaxis.set_major_formatter(FuncFormatter(seconds_to_minutes_formatter))
            plt.plot(rate_times, breathing_rates, 'b-o')
            plt.vlines(min_time, ymin=0, ymax=min_rate, colors='b', linestyle='--')
            plt.vlines(max_time, ymin=0, ymax=max_rate, colors='r', linestyle='--')

            plt.annotate(f'Min: {min_time:.2f}s, {min_rate:.2f} bpm', xy=(min_time, min_rate),
                         xytext=(min_time, min_rate + (max(breathing_rates) - min(breathing_rates)) * 0.05),
                         arrowprops=dict(facecolor='blue', shrink=0.025),
                         fontsize=10, color='blue')
            plt.annotate(f'Max: {max_time:.2f}s, {max_rate:.2f} bpm', xy=(max_time, max_rate),
                         xytext=(max_time, max_rate + (max(breathing_rates) - min(breathing_rates)) * 0.05),
                         arrowprops=dict(facecolor='red', shrink=0.025),
                         fontsize=10, color='red')

            plt.xlabel('Время')
            plt.ylabel('Частота дыхания (вдохов в минуту)')
            plt.title('Частота дыхания')
            plt.grid(True)
            plt.ylim(bottom=0)
            plt.savefig(os.path.join(output_dir, 'частота_дыхания.png'))
            #plt.show()
        else:
            fn("Not enough peaks detected to estimate breathing rate.")
    else:
        fn("Insufficient data for breathing rate analysis.")


    if gaze_x and gaze_y and len(gaze_x) == len(timestamps) and len(gaze_y) == len(timestamps):
        delta_x = np.diff(np.array(gaze_x))
        delta_y = np.diff(np.array(gaze_y))
        speeds = np.sqrt(delta_x ** 2 + delta_y ** 2)   # movement vectors

        window_size = int(fps)  # smoothing
        smoothed_speeds = np.convolve(speeds, np.ones(window_size) / window_size, mode='same')

        min_index = np.argmin(smoothed_speeds)
        max_index = np.argmax(smoothed_speeds)
        min_time = timestamps[min_index + 1]
        max_time = timestamps[max_index + 1]
        min_speed = smoothed_speeds[min_index]
        max_speed = smoothed_speeds[max_index]

        plt.figure(figsize=(15, 6), dpi=150)
        num_ticks = min(int(len(timestamps)/fps), 30)
        plt.xticks(np.linspace(timestamps[1], timestamps[-1], num_ticks), rotation=45)
        plt.gca().xaxis.set_major_formatter(FuncFormatter(seconds_to_minutes_formatter))
        plt.plot(timestamps[1:], smoothed_speeds, 'g')  # starting with +1 frame because of diffs
        plt.vlines(min_time, ymin=0, ymax=min(smoothed_speeds), colors='b', linestyle='--')
        plt.vlines(max_time, ymin=0, ymax=max(smoothed_speeds), colors='r', linestyle='--')
        plt.annotate(f'Min: {min_time:.2f}s, {min_speed:.2f}', xy=(min_time, min_speed),
                     xytext=(min_time, min_speed + (max(smoothed_speeds) - min(smoothed_speeds)) * 0.05),
                     arrowprops=dict(facecolor='blue', shrink=0.025),
                     fontsize=10, color='blue')
        plt.annotate(f'Max: {max_time:.2f}s, {max_speed:.2f}', xy=(max_time, max_speed),
                     xytext=(max_time, max_speed + (max(smoothed_speeds) - min(smoothed_speeds)) * 0.05),
                     arrowprops=dict(facecolor='red', shrink=0.025),
                     fontsize=10, color='red')
        plt.ylim(bottom=0)
        plt.xlabel("Время")
        plt.ylabel("Относительная скорость")
        plt.title("Интенсивность изменения взгляда (скользящее среднее по 1 сек.)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'интенсивность_изменения_взгляда.png'))
        plt.show()
    else:
        fn("Ошибка: информация об изменении взгляда отсутствует или не соответствует меткам времени")

    fn("###\nЗакончена оценка частоты дыхания и изменения взгляда\n###")

# Example usage
if __name__ == "__main__":
    analyze_breathing("../vid_1.mp4", "test", time_window=2)