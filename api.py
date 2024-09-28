# api.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import shutil
from fastapi import FastAPI, Request, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import faiss
from feature_extractor import FeatureExtractor
from database_manager import DatabaseManager
from database_builder import build_database
import yt_dlp
import uvicorn
import webbrowser
import logging
from uuid import uuid4

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Разрешение CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = FeatureExtractor()
db_manager = DatabaseManager()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_video(request: Request, video_url: str = Form(None), file: UploadFile = File(None), frame_step: int = Form(5)):
    try:
        temp_video_path = None
        
        logger.info(f"Received request with video_url: {video_url}, file: {file.filename if file else 'None'}, frame_step: {frame_step}")

        if video_url:
            # Если передан URL, пробуем скачать видео
            try:
                logger.info(f"Attempting to download video from URL: {video_url}")
                ydl_opts = {
                    'outtmpl': 'temp/%(id)s.%(ext)s',
                    'format': 'best',
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(video_url, download=True)
                    temp_video_path = os.path.join("temp", f"{info_dict['id']}.{info_dict['ext']}")
                    logger.info(f"Video downloaded to: {temp_video_path}")
            except Exception as e:
                logger.error(f"Error downloading video: {e}")
                raise HTTPException(status_code=500, detail=f"Ошибка при загрузке видео по ссылке: {e}")

        elif file:
            # Если загружен файл, сохраняем его в temp
            try:
                if not os.path.exists("temp"):
                    os.makedirs("temp")
                temp_video_path = os.path.join("temp", file.filename)
                with open(temp_video_path, "wb") as buffer:
                    buffer.write(await file.read())
                logger.info(f"File uploaded and saved to: {temp_video_path}")
            except Exception as e:
                logger.error(f"Error saving uploaded file: {e}")
                raise HTTPException(status_code=500, detail=f"Ошибка при сохранении загруженного файла: {e}")
        else:
            raise HTTPException(status_code=400, detail="Необходимо передать видео через URL или загрузить файл.")

        # Извлечение признаков из скачанного или загруженного видео
        try:
            logger.info(f"Starting feature extraction from: {temp_video_path}")
            features = extractor.extract(temp_video_path, frame_step=frame_step)
            if features is None:
                logger.error(f"Feature extraction failed for video: {temp_video_path}")
                os.remove(temp_video_path)
                raise HTTPException(status_code=400, detail="Не удалось извлечь признаки из видео.")
        except Exception as e:
            logger.error(f"Error during feature extraction: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка при извлечении признаков из видео: {e}")

        # Поиск дубликатов
        try:
            logger.info(f"Starting duplicate search for video: {temp_video_path}")
            db_features, video_ids = db_manager.get_all_features()
            if len(db_features) == 0:
                is_duplicate = False
                closest_video_id = None
                distance = None
            else:
                d = features.shape[0]
                index = faiss.IndexFlatL2(d)
                index.add(db_features.astype('float32'))
                D, I = index.search(np.array([features]).astype('float32'), k=1)
                distance = D[0][0]  # Получаем расстояние до ближайшего видео
                closest_video_id = video_ids[I[0][0]]
                threshold = 0.5
                is_duplicate = bool(distance < threshold)
                distance = float(distance) if distance is not None else "Не определено"  # Убедимся, что расстояние установлено

            # Формирование результата
            if is_duplicate:
                message = f"Видео является дубликатом видео с ID: {closest_video_id}"
                logger.info(f"Duplicate found: {closest_video_id} with distance {distance}")
                os.remove(temp_video_path)  # Удаляем временный файл
            else:
                message = ("Видео уникально и не является дубликатом. "
                           "Видео добавлено в базу данных. База данных обновлена.")
                video_id = os.path.splitext(file.filename)[0] if file else info_dict['id']

                # Добавляем проверку на существование файла
                destination_path = os.path.join("videos", os.path.basename(temp_video_path))
                if os.path.exists(destination_path):
                    unique_suffix = uuid4().hex[:8]  # Генерируем уникальный суффикс
                    destination_path = os.path.join("videos", f"{os.path.splitext(os.path.basename(temp_video_path))[0]}_{unique_suffix}{os.path.splitext(temp_video_path)[1]}")
                    logger.info(f"File already exists. Renamed to: {destination_path}")

                shutil.move(temp_video_path, destination_path)
                build_database('videos')  # Пересобираем базу данных
                logger.info(f"Video {video_id} added to database as {destination_path}")

            result = {
                "filename": file.filename if file else info_dict['title'],
                "is_duplicate": is_duplicate,
                "closest_video_id": closest_video_id if closest_video_id else "Дубликат не найден",
                "distance": distance if isinstance(distance, float) else "Не определено",  # Проверка корректности значения
                "message": message
            }

            return templates.TemplateResponse("index.html", {"request": request, "result": result})
        except Exception as e:
            logger.error(f"Error during duplicate search or database update: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка при поиске дубликатов: {e}")

    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise he

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера.")
    

# Запуск приложения с автоматическим открытием браузера
if __name__ == "__main__":
    import threading

    threading.Timer(1.5, lambda: webbrowser.open_new("http://127.0.0.1:8000")).start()
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
