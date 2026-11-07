@echo off
setlocal EnableDelayedExpansion

REM ---------------------------------------------------------------
REM  COLMAP Point Editor launcher
REM  Activates Anaconda, installs any missing Python packages, then
REM  launches the editor.
REM ---------------------------------------------------------------

set "SCRIPT=C:\Users\%USERNAME%\.lichtfeld\plugins\Pointnuker_SOR\scripts\PyVista_ColmapBIN-CRS_Pyside.py"

REM Force pyvistaqt to use PySide6 (avoids clashes if PyQt5/PyQt6 is also installed)
set "QT_API=pyside6"

REM Activate Anaconda base environment
call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Could not activate Anaconda at %USERPROFILE%\anaconda3
    pause
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo [ERROR] Script not found:
    echo   %SCRIPT%
    pause
    exit /b 1
)

REM Check each required module; collect the pip names of any that are missing.
REM Format:  import_name=pip_package_name
set "MISSING="
for %%P in (numpy=numpy scipy=scipy vtk=vtk pyvista=pyvista PySide6=PySide6 pyvistaqt=pyvistaqt) do (
    for /f "tokens=1,2 delims==" %%A in ("%%P") do (
        python -c "import %%A" >nul 2>&1
        if errorlevel 1 set "MISSING=!MISSING! %%B"
    )
)

if defined MISSING (
    echo Installing missing packages:%MISSING%
    python -m pip install%MISSING%
    if errorlevel 1 (
        echo [ERROR] Package installation failed.
        pause
        exit /b 1
    )
)

python "%SCRIPT%"
if errorlevel 1 (
    echo.
    echo [ERROR] The editor exited with an error.
    pause
)

endlocal
