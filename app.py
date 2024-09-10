import json

from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
import os
import shutil
from pathlib import Path
import zipfile
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List
import uuid
import secrets

import config
from utils.breathing_and_gaze import analyze_breathing
from utils.heartrate import process_video_with_pyvhr

app = FastAPI()
templates = Jinja2Templates(directory="templates")
security = HTTPBasic()

USERNAME = config.LOGIN
PASSWORD = config.PASSWORD

processing_flag = False
current_video = None
current_video_info = None
log_messages = []
results_dir = "./results"
uploads_dir = "./uploads"
archives_dir = "./archives"
error_log_path = Path("./errors.json")
video_queue = asyncio.Queue()

progress_data = {
    "breathing_and_gaze": {"progress": 0, "status": "Ожидание...", "error": None},
    "heartrate": {"progress": 0, "status": "Ожидание...", "error": None}
}

os.makedirs(results_dir, exist_ok=True)
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(archives_dir, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=2)


def log(message):
    log_messages.append(message)
    print(message)


def update_breathing_progress(percentage, status):
    progress_data["breathing_and_gaze"]["progress"] = percentage
    progress_data["breathing_and_gaze"]["status"] = status


def update_heartrate_progress(percentage, status):
    progress_data["heartrate"]["progress"] = percentage
    progress_data["heartrate"]["status"] = status


def write_error_log(video_name, uploaded_at, breathing_status, breathing_error, heartrate_status, heartrate_error):
    error_entry = {
        "video_name": video_name,
        "uploaded_at": uploaded_at.strftime('%Y-%m-%d %H:%M:%S'),
        "breathing_and_gaze": {
            "status": breathing_status,
            "error": breathing_error
        },
        "heartrate": {
            "status": heartrate_status,
            "error": heartrate_error
        }
    }

    if error_log_path.exists():
        with open(error_log_path, "r", encoding="utf-8") as f:
            error_log = json.load(f)
    else:
        error_log = []

    error_log.append(error_entry)

    with open(error_log_path, "w", encoding="utf-8") as f:
        json.dump(error_log, f, ensure_ascii=False, indent=4)


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)

    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Неверные учетные данные", headers={"WWW-Authenticate": "Basic"})


@app.get("/", response_class=JSONResponse)
async def main_page(request: Request, credentials: HTTPBasicCredentials = Depends(authenticate)):
    archives = [
        {"name": archive.name,
         "timestamp": datetime.fromtimestamp(archive.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}
        for archive in sorted(Path(archives_dir).iterdir(), key=os.path.getmtime, reverse=True)
    ]

    current_queue = list(video_queue._queue)
    current_processing = current_video

    errors = []
    if error_log_path.exists():
        with open(error_log_path, "r", encoding="utf-8") as f:
            errors = json.load(f)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "processing_flag": processing_flag,
        "current_video": current_processing,
        "log_messages": "<br>".join(log_messages),
        "archives": archives,
        "queue": current_queue,
        "errors": errors
    })


@app.post("/upload_video")
async def upload_video(files: List[UploadFile] = File(...),
                       background_tasks: BackgroundTasks = BackgroundTasks(),
                       credentials: HTTPBasicCredentials = Depends(authenticate)):
    global processing_flag, current_video, log_messages

    for video in files:
        video_filename = video.filename
        video_name, video_extension = os.path.splitext(video_filename)

        unique_id = datetime.now().strftime('%Y%m%d_%H%M%S') + "_" + str(uuid.uuid4())[:8]
        new_video_filename = f"{video_name}_{unique_id}{video_extension}"

        video_path = Path(uploads_dir) / new_video_filename

        with video_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        log_messages = []
        log(f"Видео {new_video_filename} загружено и добавлено в очередь.")

        await video_queue.put({"path": video_path, "name": new_video_filename, "uploaded_at": datetime.now()})

    if not processing_flag:
        asyncio.create_task(process_video_queue())

    return {"message": f"{len(files)} видео загружены и добавлены в очередь."}


async def process_video_queue():
    """Основной цикл"""
    global processing_flag, current_video, current_video_info

    while not video_queue.empty():
        video_data = await video_queue.get()
        video_path = video_data["path"]
        video_name = video_data["name"]
        uploaded_at = video_data["uploaded_at"]
        current_video = video_name
        processing_flag = True

        progress_data["breathing_and_gaze"]["error"] = None
        progress_data["heartrate"]["error"] = None

        try:
            output_dir = Path(results_dir) / f"{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir.mkdir(parents=True, exist_ok=True)

            log(f"Начинаем обработку видео: {video_name}")

            current_video_info = get_video_info(video_path)

            if current_video_info is None:
                log(f"Ошибка! Не удалось открыть видеофайл: {video_path}")
                continue

            log("Запускаем анализ дыхания и взгляда и сердцебиения...")

            async def run_breathing_analysis():
                return await asyncio.get_event_loop().run_in_executor(
                    executor, analyze_breathing, str(video_path), str(output_dir), 2, update_breathing_progress
                )

            async def run_heartrate_analysis():
                return await asyncio.get_event_loop().run_in_executor(
                    executor, process_video_with_pyvhr, str(video_path), 'patches', 'median', 'cpu_OMIT', 0, str(output_dir), update_heartrate_progress
                )

            results = await asyncio.gather(
                run_breathing_analysis(),
                run_heartrate_analysis(),
                return_exceptions=True
            )

            breathing_result, heartrate_result = results
            breathing_status = "success" if not isinstance(breathing_result, Exception) else "fail"
            heartrate_status = "success" if not isinstance(heartrate_result, Exception) else "fail"

            if isinstance(breathing_result, Exception):
                log(f"Ошибка при анализе дыхания: {breathing_result}")
                progress_data["breathing_and_gaze"]["error"] = str(breathing_result)

            if isinstance(heartrate_result, Exception):
                log(f"Ошибка при анализе сердцебиения: {heartrate_result}")
                progress_data["heartrate"]["error"] = str(heartrate_result)

            if isinstance(breathing_result, Exception) or isinstance(heartrate_result, Exception):
                write_error_log(
                    video_name,
                    uploaded_at,
                    breathing_status,
                    str(breathing_result) if isinstance(breathing_result, Exception) else None,
                    heartrate_status,
                    str(heartrate_result) if isinstance(heartrate_result, Exception) else None
                )

            if breathing_status == "success" or heartrate_status == "success":
                archive_results(output_dir, video_name)

        except Exception as e:
            log(f"Ошибка при обработке видео: {e}")
            progress_data["breathing_and_gaze"]["error"] = str(e)
            progress_data["heartrate"]["error"] = str(e)

        finally:
            processing_flag = False
            current_video = None
            try:
                os.remove(video_path)
            except OSError as e:
                log(f"Ошибка при удалении видеофайла: {e}")

        await asyncio.sleep(1)


@app.post("/remove_from_queue/{filename}")
async def remove_from_queue(filename: str):
    """Удаление видео из очереди на обработку"""
    global video_queue
    video_queue = [video for video in video_queue._queue if video["name"] != filename]
    return {"message": f"Видео {filename} удалено из очереди."}


@app.delete("/delete_archive/{archive_name}")
async def delete_archive(archive_name: str):
    """Удаленгие архива с результатами"""
    archive_path = Path(archives_dir) / archive_name
    if archive_path.exists():
        os.remove(archive_path)
        return {"message": "Архив удален."}
    return JSONResponse({"message": "Файл не найден."}, status_code=404)


@app.get("/progress", response_class=JSONResponse)
async def get_progress():
    """Текущий прогресс обработки видео"""
    return {
        "breathing_and_gaze": progress_data["breathing_and_gaze"],
        "heartrate": progress_data["heartrate"],
        "video_info": current_video_info  # Используем сохраненную информацию о видео
    }


def get_video_info(video_path):
    """Основная информация о видео (перед началом обработки)"""
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
        "name": video_path.name,
        "duration": duration,
        "resolution": f"{width}x{height}",
        "fps": fps
    }


@app.on_event("startup")
async def initialize_queue():
    """Автоматическое добавление видео из uploads в очередь при запуске"""
    files = os.listdir(uploads_dir)
    for file_name in files:
        video_path = Path(uploads_dir) / file_name
        if video_path.is_file():
            await video_queue.put({"path": video_path, "name": file_name, "uploaded_at": datetime.now()})
    if not video_queue.empty():
        asyncio.create_task(process_video_queue())  # Асинхронный запуск обработки очереди


def archive_results(output_dir, video_name):
    """Архивация результатов для дальнейшего скачивания. Если папка пуста, просто удаляем её"""
    if not any(output_dir.iterdir()):
        log("Папка результатов пуста, удаляем её.")
        output_dir.rmdir()
    else:
        # Разделяем имя видео и добавляем дату и время перед расширением
        archive_name, _ = os.path.splitext(video_name)
        archive_name = f"{archive_name}.zip"

        archive_path = Path(archives_dir) / archive_name
        with zipfile.ZipFile(archive_path, "w") as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    file_path = Path(root) / file
                    zipf.write(file_path, arcname=file_path.relative_to(output_dir))
        log(f"Результаты архивированы: {archive_path.name}")
        shutil.rmtree(output_dir)
        log(f"Папка {output_dir} удалена.")


@app.get("/download/{archive_name}")
async def download_archive(archive_name: str):
    """Скачивание архива"""
    archive_path = Path(archives_dir) / archive_name
    if archive_path.exists():
        return FileResponse(archive_path, media_type='application/zip', filename=archive_name)
    return JSONResponse({"message": "Файл не найден."})

@app.get("/get_queue", response_class=JSONResponse)
async def get_queue():
    """Возвращает список видео в очереди"""
    queue_list = [
        {"name": video["name"], "uploaded_at": video["uploaded_at"].strftime('%Y-%m-%d %H:%M:%S')}
        for video in video_queue._queue  # Доступ к очереди через _queue
    ]
    current_video_data = {
        "current_video": current_video,
        "video_info": current_video_info
    }
    return {"queue": queue_list, "current_video": current_video_data}


@app.get("/get_archives", response_class=JSONResponse)
async def get_archives():
    """Возвращает список архивов с результатами"""
    archives = [
        {"name": archive.name, "timestamp": datetime.fromtimestamp(archive.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}
        for archive in sorted(Path(archives_dir).iterdir(), key=os.path.getmtime, reverse=True)
    ]
    return {"archives": archives}

@app.get("/get_errors", response_class=JSONResponse)
async def get_errors():
    """Возвращает список ошибок из файла errors.json"""
    if error_log_path.exists():
        with open(error_log_path, "r", encoding="utf-8") as f:
            errors = json.load(f)
        return {"errors": errors}
    return {"errors": []}
