@echo off
REM Abre a pagina do C. elegans. Nada treina sozinho: use os botoes da pagina.
cd /d "%~dp0"
start "" http://127.0.0.1:8770
set HF_HOME=D:\Mateus\.hf-cache
"D:\Mateus\Laya\.venv\Scripts\python.exe" servidor.py
pause
