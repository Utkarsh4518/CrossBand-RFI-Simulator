@echo off
REM Build Project_Equations.tex from repo root. Requires pdflatex on PATH.
cd /d "%~dp0.."
if not exist "build" mkdir build
pdflatex -output-directory=build -interaction=nonstopmode Project_Equations.tex
if %ERRORLEVEL% equ 0 (
    echo.
    echo PDF: build\Project_Equations.pdf
) else (
    exit /b 1
)
