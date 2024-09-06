from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import os
import shutil
from pathlib import Path
import zipfile
import asyncio
from datetime import datetime
import re

from utils.breathing_and_gaze import analyze_breathing
from utils.heartrate import process_video_with_pyvhr
from utils.utils import seconds_to_minutes_formatter



app = FastAPI()

processing_flag = False
current_video = None
log_messages = []
results_dir = "./results"
uploads_dir = "./uploads"
archives_dir = "./archives"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(archives_dir, exist_ok=True)


def log(message):
    log_messages.append(message)
    print(message)


@app.get("/", response_class=HTMLResponse)
async def main_page():
    if processing_flag:
        log_content = "<br>".join(log_messages)
        video_processing_section = f"""
        <h2>Идёт обработка видео: {current_video}</h2>
        <div id="log-section">
            <pre>{log_content}</pre>
        </div>
        """
    else:
        video_processing_section = """
        <h2>Загрузите видео для анализа</h2>
        <form id="upload-form" enctype="multipart/form-data" method="post">
            <input id="file-input" name="video" type="file" accept="video/mp4">
            <input type="submit" value="Загрузить видео">
        </form>
        """

    archives = sorted(Path(archives_dir).iterdir(), key=os.path.getmtime, reverse=True)
    if archives:
        archive_list = "".join(
            [f"<li>{archive.name} - {datetime.fromtimestamp(archive.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')} "
             f"<a href='/download/{archive.name}'>Скачать .ZIP</a></li>" for archive in archives]
        )
        download_section = f"<ul>{archive_list}</ul>"
    else:
        download_section = "<p>Нет доступных результатов.</p>"

    html_content = f"""
    <html>
    <body>
        <h1>Анализ видео</h1>
        {video_processing_section}
        <hr>
        <h1>Загрузка результатов</h1>
        {download_section}
        <script>
            setInterval(async function() {{
                const response = await fetch("/status");
                const data = await response.json();
                if (data.processing) {{
                    document.getElementById("log-section").innerHTML = data.html;
                }}
            }}, 2000);

            document.getElementById("upload-form")?.addEventListener("submit", async function(event) {{
                event.preventDefault();
                const formData = new FormData();
                const fileInput = document.getElementById("file-input");
                if (fileInput.files.length > 0) {{
                    formData.append("video", fileInput.files[0]);
                    const response = await fetch("/upload_video", {{
                        method: "POST",
                        body: formData
                    }});
                    const data = await response.json();
                    alert(data.message);
                    window.location.reload();
                }}
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/status", response_class=JSONResponse)
async def status():
    if processing_flag:
        log_content = "<br>".join(log_messages)
        video_processing_section = f"""
        <h2>Идёт обработка видео: {current_video}</h2>
        <pre>{log_content}</pre>
        """
    else:
        video_processing_section = """
        <h2>Загрузите видео для анализа</h2>
        <form id="upload-form" enctype="multipart/form-data" method="post">
            <input id="file-input" name="video" type="file" accept="video/mp4">
            <input type="submit" value="Загрузить видео">
        </form>
        """

    return {
        "processing": processing_flag,
        "html": video_processing_section
    }


@app.post("/upload_video")
async def upload_video(video: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    global processing_flag, current_video, log_messages

    if processing_flag:
        return JSONResponse({"message": "Видео уже обрабатывается. Подождите завершения."})

    video_filename = video.filename
    video_path = Path(uploads_dir) / video_filename
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    log_messages = []
    current_video = video_filename
    processing_flag = True

    background_tasks.add_task(process_video, video_path)

    return {"message": f"Видео {video_filename} загружено. Обработка началась."}


async def process_video(video_path: Path):
    global processing_flag

    try:
        video_name = video_path.stem
        output_dir = Path(results_dir) / f"{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        log(f"Начинаем обработку видео: {video_path.name}")

        # Получаем информацию о видео
        video_info = get_video_info(video_path)

        if video_info is None:
            log(f"Ошибка: не удалось открыть видео {video_path}")
            return

        log(f"Длительность: {video_info['duration']:.2f} секунд ({seconds_to_minutes_formatter(video_info['duration'], None)})")
        log(f"Разрешение: {video_info['resolution']}")
        log(f"FPS: {video_info['fps']}")

        log("Запускаем анализ дыхания и взгляда...")
        analyze_breathing(str(video_path), str(output_dir), fn=log)

        log("Запускаем анализ сердцебиения с pyVHR...")
        process_video_with_pyvhr(str(video_path), 'patches', 'median', 'cpu_OMIT', 0, str(output_dir), fn=log)


        # Архивация результатов
        if not any(output_dir.iterdir()):
            log("Папка результатов пуста, удаляем её.")
            output_dir.rmdir()
        else:
            archive_name = re.sub(r'[^\w\d-]', '_', output_dir.stem) + ".zip"
            archive_path = Path(archives_dir) / archive_name
            with zipfile.ZipFile(archive_path, "w") as zipf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = Path(root) / file
                        zipf.write(file_path, arcname=file_path.relative_to(output_dir))
            log(f"Результаты архивированы: {archive_path.name}")

            shutil.rmtree(output_dir)
            log(f"Папка {output_dir} удалена.")
        log("Обработка завершена.")
    except Exception as e:
        log(f"Ошибка при обработке: {e}")
    finally:
        processing_flag = False
        os.remove(video_path)


def get_video_info(video_path):
    import cv2

    video = cv2.VideoCapture(str(video_path))

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


@app.get("/download/{archive_name}")
async def download_archive(archive_name: str):
    archive_path = Path(archives_dir) / archive_name
    if archive_path.exists():
        return FileResponse(archive_path, media_type='application/zip', filename=archive_name)
    return JSONResponse({"message": "Файл не найден."})