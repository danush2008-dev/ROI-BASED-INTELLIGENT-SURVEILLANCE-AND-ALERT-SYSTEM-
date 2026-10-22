@echo off
echo ============================================
echo  ROI-Based Intelligent Surveillance System
echo  Video Runner
echo ============================================
echo.
echo Select a video to run:
echo   1. video 1.mp4
echo   2. video 2.mp4
echo   3. video 3.mp4
echo   4. input vedio for testing.mp4 (original)
echo   5. Run all 3 new videos (one by one)
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto video1
if "%choice%"=="2" goto video2
if "%choice%"=="3" goto video3
if "%choice%"=="4" goto original
if "%choice%"=="5" goto all
echo Invalid choice.
goto end

:video1
echo.
echo Running video 1.mp4 ...
python phy.py --source "video 1.mp4"
goto end

:video2
echo.
echo Running video 2.mp4 ...
python phy.py --source "video 2.mp4"
goto end

:video3
echo.
echo Running video 3.mp4 ...
python phy.py --source "video 3.mp4"
goto end

:original
echo.
echo Running input vedio for testing.mp4 ...
python phy.py
goto end

:all
echo.
echo === Running video 1.mp4 ===
python phy.py --source "video 1.mp4"
echo.
echo === Running video 2.mp4 ===
python phy.py --source "video 2.mp4"
echo.
echo === Running video 3.mp4 ===
python phy.py --source "video 3.mp4"
goto end

:end
echo.
echo Done.
pause
