import os
import argparse
import cv2

print("Инициализирую зависимости...")

from utils.breathing_and_gaze import analyze_breathing
from utils.heartrate import process_video_with_pyvhr
from utils.utils import seconds_to_minutes_formatter

# params
ROI_APPROACH = 'patches'    # 'patches' or 'holistic'
BPM_EST = 'median'          # BPM final estimate, if patches choose 'medians' or 'clustering'
ESTIMATOR_IX = 0            # estimator for BVP (-1 for average value)
PYVHR_METHOD = 'cpu_OMIT'   # DIFFERENT SUPPORTED METHODS:
                            # cpu_CHROM, cupy_CHROM, torch_CHROM, cpu_LGI, cpu_POS, cupy_POS, cpu_PBV, cpu_PCA, cpu_GREEN, cpu_OMIT, cpu_ICA, cpu_SSR
                            # source 1 (README.md): https://github.com/phuselab/pyVHR/tree/master?tab=readme-ov-file#methods
                            # source 2: https://github.com/phuselab/pyVHR/blob/master/notebooks/pyVHR_demo.ipynb
                            # OMIT is the latest method as for now

def get_video_info(video_path):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        return None

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps

    video.release()
    return {
        "duration": duration,
        "resolution": f"{width}x{height}",
        "fps": fps
    }


def main(video_path=None):
    while True:
        arg_provided = True
        if not video_path:
            video_path = input("Введите путь к видеофайлу для анализа (с расширением): ")
            arg_provided = False

        if not os.path.isfile(video_path):
            print("Файл не существует. Пожалуйста, попробуйте снова")
            if arg_provided:
                return
            video_path = None
            continue

        if not video_path.lower().endswith('.mp4'):
            print("Видео должно иметь расширение mp4. Пожалуйста, повторите")
            if arg_provided:
                return
            video_path = None
            continue

        video_info = get_video_info(video_path)

        if video_info is None:
            print("Не удалось открыть видео. Пожалуйста, выберите другой файл")
            if arg_provided:
                return
            video_path = None
            continue

        base_output_dir = "./results/"
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(base_output_dir, video_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            count = 1
            while os.path.exists(output_dir + f"_{count}"):
                count += 1
            output_dir = output_dir + f"_{count}"
            os.makedirs(output_dir)

        output_dir = os.path.abspath(output_dir)
        video_path = os.path.abspath(video_path)

        print(f"Директория для сохранения результатов: {output_dir}/".replace("\\", "/"))

        print(f"Длительность: {video_info['duration']:.2f} секунд "
              f"({seconds_to_minutes_formatter(video_info['duration'], None)})")
        print(f"Разрешение: {video_info['resolution']}")
        print(f"FPS: {video_info['fps']}")

        print(video_path)

        analyze_breathing(video_path, output_dir)
        process_video_with_pyvhr(video_path, ROI_APPROACH, BPM_EST, PYVHR_METHOD, ESTIMATOR_IX, output_dir)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vidMetrics Tools")
    parser.add_argument("video_path", type=str, nargs='?', help="Path to the .mp4 video file")

    args = parser.parse_args()

    main(args.video_path)