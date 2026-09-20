@echo off
setlocal

if not exist .venv (
    echo Creating virtual environment...
    py -3.11 -m venv .venv
)

call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Running CLIP defense evaluation...
python -m src.evaluate --max-eval-samples 500 --batch-size 16 --steps 10

echo Done. Metrics saved in results\metrics\evaluation.json
endlocal
