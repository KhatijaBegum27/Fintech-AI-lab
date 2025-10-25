@echo off
color 0A
echo ===============================================
echo     CREDIT RISK ML APP - AUTO LAUNCHER
echo ===============================================
echo.

:: Step 1: Create or activate virtual environment
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate

:: Step 2: Install dependencies (if not installed)
echo Installing required libraries...
pip install --quiet pandas numpy matplotlib seaborn scikit-learn xgboost fastapi uvicorn joblib requests streamlit

:: Step 3: Start FastAPI backend
echo Starting FastAPI backend...
start cmd /k "uvicorn Model_API:app --reload"

:: Step 4: Wait 5 seconds, then start Streamlit frontend
timeout /t 5 >nul
echo Starting Streamlit frontend...
start cmd /k "streamlit run app_streamlit.py"

echo ===============================================
echo   Application running locally!
echo   API:      http://127.0.0.1:8000/docs
echo   Streamlit: http://localhost:8501
echo ===============================================
pause
