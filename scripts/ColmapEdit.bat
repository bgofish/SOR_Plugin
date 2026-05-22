@echo off

REM Activate Anaconda base environment
call %USERPROFILE%\anaconda3\Scripts\activate.bat

python "C:\Users\%username%\.lichtfeld\plugins\Pointnuker_SOR\scripts\PyVista_ColmapBIN-CRS_Pyside.py"