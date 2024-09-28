@echo off
REM Устанавливаем переменную PATH так, чтобы использовался Python из папки Python312
set "PATH=%cd%\Python312;%PATH%"

REM Проверяем, существует ли виртуальное окружение
if not exist "venv\Scripts\activate.bat" (
    REM Создаем виртуальное окружение с использованием локального Python
    .\Python312\python.exe -m venv venv
)

REM Активируем виртуальное окружение
call venv\Scripts\activate.bat

REM Устанавливаем зависимости (если необходимо)
pip install -r requirements.txt

REM Запускаем приложение
python api.py
