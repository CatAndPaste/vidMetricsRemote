import os
import numpy as np
import pyVHR as vhr
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pyVHR.analysis.pipeline import Pipeline
import matplotlib.pyplot as plt


def seconds_to_minutes_formatter(x, pos):
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f'{minutes:02d}:{seconds:02d}'


def process_video_with_pyvhr(video_file, roi_approach, bpm_est, method, estimator_index, output_dir, fn=print):
    fn("Начинаю обработку с помощью pyVHR (BVP+BPM)")
    wsize = 5  # seconds of video processed (with overlapping) for each estimate
    fps = vhr.extraction.get_fps(video_file)

    pipe = Pipeline()

    # starting pipeline
    bvps, timesES, bpmES = pipe.run_on_video(video_file,
                                             winsize=wsize,
                                             roi_method='convexhull',
                                             roi_approach=roi_approach,
                                             method=method,
                                             estimate=bpm_est,
                                             patch_size=40,
                                             RGB_LOW_HIGH_TH=(5, 230),
                                             Skin_LOW_HIGH_TH=(5, 230),
                                             pre_filt=True,
                                             post_filt=True,
                                             cuda=True,
                                             verb=True)

    fn("Сохраняю результаты...")
    save_dir = os.path.join(output_dir)
    os.makedirs(save_dir, exist_ok=True)
    bvp_windows_path = os.path.join(save_dir, 'bvp_time_windows')
    os.makedirs(bvp_windows_path, exist_ok=True)

    # plots
    step_size = 1   # in seconds, step of time windows: win 1 starts at 0, win 2 starts at step_size,
                    # win 3 - at step_size * 2, etc.

    # Choosing specific estimator for final BVP plot
    def get_bvps_for_plot(bvps, estimator_index):
        if estimator_index == -1:
            return [np.mean(bvp_window, axis=0) for bvp_window in bvps]
        elif estimator_index >= 0 and estimator_index < bvps[0].shape[0]:
            return [bvp_window[estimator_index] for bvp_window in bvps]
        else:
            raise ValueError(f"Estimator {estimator_index} not found in the data.")

    # Combining time windows to get final BVP plot
    def combine_bvps(bvps_aggregated, wsize, step_size, fps):
        frames_per_window = bvps_aggregated[0].shape[0]
        total_length = frames_per_window + (len(bvps_aggregated) - 1) * step_size * fps
        combined_bvps = np.zeros(int(total_length))
        time_vector = np.linspace(0, len(combined_bvps) / fps, len(combined_bvps))

        for i in range(len(bvps_aggregated)):
            start_idx = int(i * step_size * fps)
            end_idx = start_idx + frames_per_window

            current_window = bvps_aggregated[i]

            # last window might be shorter than others
            if end_idx > len(combined_bvps):
                end_idx = len(combined_bvps)
                current_window = current_window[:end_idx - start_idx]

            combined_bvps[start_idx:end_idx] += current_window

            # Handling overlapping
            if i > 0:
                overlap_length = min(int(step_size * fps), end_idx - start_idx)
                combined_bvps[start_idx:start_idx + overlap_length] /= 2

        return combined_bvps, time_vector

    bvps_aggregated = get_bvps_for_plot(bvps, estimator_index)
    combined_bvps, time_vector = combine_bvps(bvps_aggregated, wsize, step_size, fps)

    # Plotting BVP Signal
    plt.figure(figsize=(15, 8), dpi=150)
    num_ticks = min(len(time_vector), 30)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(seconds_to_minutes_formatter))
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(MaxNLocator(num_ticks))
    plt.axhline(0, color='black', linewidth=1.5)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['bottom'].set_color('black')
    plt.plot(time_vector, combined_bvps, label=f'BVP Signal (Estimator {estimator_index})')

    min_bvp = np.min(combined_bvps)
    max_bvp = np.max(combined_bvps)
    min_time = time_vector[np.argmin(combined_bvps)]
    max_time = time_vector[np.argmax(combined_bvps)]
    plt.vlines(min_time, ymin=min(0, min_bvp), ymax=max(0, min_bvp), colors='blue', linestyle='--')
    plt.vlines(max_time, ymin=min(0, max_bvp), ymax=max(0, max_bvp), colors='red', linestyle='--')
    plt.annotate(f'Min: {min_time:.2f}s, {min_bvp:.2f}', xy=(min_time, min_bvp),
                 xytext=(min_time, min_bvp - 0.5),
                 arrowprops=dict(facecolor='blue', shrink=0.05),
                 fontsize=10, color='blue')
    plt.annotate(f'Max: {max_time:.2f}s, {max_bvp:.2f}', xy=(max_time, max_bvp),
                 xytext=(max_time, max_bvp + 0.5),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=10, color='red')
    plt.title('Кровеносное давление (BVP)')
    plt.xlabel('Время')
    plt.ylabel('Значение BVP-сигнала')
    plt.legend()
    plt.grid(True)
    bvp_plot_path = os.path.join(save_dir, 'pyvhr_кровеносное_давление.png')
    plt.savefig(bvp_plot_path)
    #plt.show()

    # Plotting BPM
    time_bpm = timesES
    bpm_values = bpmES.flatten()

    global_min_bpm = np.min(bpm_values)
    global_max_bpm = np.max(bpm_values)
    min_time_bpm = time_bpm[np.argmin(bpm_values)]
    max_time_bpm = time_bpm[np.argmax(bpm_values)]

    plt.figure(figsize=(15, 8), dpi=150)
    plt.gca().xaxis.set_major_locator(MaxNLocator(num_ticks))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(seconds_to_minutes_formatter))
    plt.xticks(rotation=45)
    plt.plot(time_bpm, bpm_values, label='BPM')
    plt.vlines(min_time_bpm, ymin=0, ymax=global_min_bpm, colors='blue', linestyle='--', label='Min BPM')
    plt.vlines(max_time_bpm, ymin=0, ymax=global_max_bpm, colors='red', linestyle='--', label='Max BPM')
    plt.scatter(min_time_bpm, global_min_bpm, color='blue', label='Min BPM')
    plt.scatter(max_time_bpm, global_max_bpm, color='red', label='Max BPM')

    plt.annotate(f'Min: {min_time_bpm:.2f}s, {global_min_bpm:.2f}', xy=(min_time_bpm, global_min_bpm),
                 xytext=(min_time_bpm, global_min_bpm - 5),
                 arrowprops=dict(facecolor='blue', shrink=0.05),
                 fontsize=10, color='blue')
    plt.annotate(f'Max: {max_time_bpm:.2f}s, {global_max_bpm:.2f}', xy=(max_time_bpm, global_max_bpm),
                 xytext=(max_time_bpm, global_max_bpm + 5),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=10, color='red')
    plt.ylim(bottom=0)
    plt.title('Частота пульса')
    plt.xlabel('Время')
    plt.ylabel('Ударов в минуту (BPM)')
    plt.legend()
    plt.grid(True)
    bpm_plot_path = os.path.join(save_dir, 'pyvhr_пульс.png')
    plt.savefig(bpm_plot_path)
    #plt.show()

    # Saving BVP plots for each time window
    for window_idx, bvp_window in enumerate(bvps):
        time_start = timesES[window_idx] - wsize / 2
        time_end = time_start + wsize
        time = np.linspace(time_start, time_end, bvp_window.shape[1])

        plt.figure(figsize=(10, 6))
        for estimator_idx, bvp_signal in enumerate(bvp_window):
            plt.plot(time, bvp_signal, label=f'Estimator {estimator_idx + 1}')

        plt.title(f'Сигналы BVP для временного окна {window_idx + 1} (с {time_start:.2f} по {time_end:.2f} с.)')
        plt.xlabel('Время')
        plt.ylabel('Сигнал BVP')
        plt.legend()
        plt.grid(True)

        window_plot_path = os.path.join(bvp_windows_path, f'bvp_window_{window_idx + 1}.png')
        plt.savefig(window_plot_path)
        plt.close()

    fn(f"Обработка завершена! Результаты сохранены в {save_dir}")