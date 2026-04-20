@echo off
chcp 65001 >nul
echo.
echo ================================================
echo   Build Consigliere di Stazione v1.0
echo   di I6502TR
echo ================================================
echo.

:: Vai alla root del progetto (cartella sopra build\)
cd /d "%~dp0\.."

:: Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRORE: Python non trovato nel PATH.
    echo Installa Python da https://www.python.org/downloads/
    echo Assicurati di spuntare "Add Python to PATH" durante l'installazione.
    pause
    exit /b 1
)

echo Python trovato:
python --version
echo.

:: Installa dipendenze
echo Installazione dipendenze progetto...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERRORE: pip install requirements.txt fallito.
    pause
    exit /b 1
)

echo Installazione PyInstaller...
pip install pyinstaller --quiet
if errorlevel 1 (
    echo ERRORE: installazione PyInstaller fallita.
    pause
    exit /b 1
)

echo.
echo Build in corso (potrebbe richiedere 1-2 minuti)...
echo.

:: Esegui il build
pyinstaller build\consigliere.spec --clean --noconfirm

echo.
if exist "dist\ConsigliereDiStazione.exe" (
    echo ================================================
    echo   BUILD COMPLETATO CON SUCCESSO!
    echo.
    echo   File prodotto:
    echo   dist\ConsigliereDiStazione.exe
    echo.
    echo   Comprimi il file in uno ZIP e distribuiscilo.
    echo ================================================
) else (
    echo ERRORE: il file exe non e' stato prodotto.
    echo Controlla i messaggi di errore sopra.
)

echo.
pause
