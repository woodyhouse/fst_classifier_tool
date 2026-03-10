@echo off
chcp 65001 >nul
pushd "%~dp0\..\.."

echo ========================================
echo DMPR-PS 自动测试
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python
    pause
    popd
    exit /b 1
)

echo [1/3] 检查依赖...
pip show torch >nul 2>&1
if errorlevel 1 (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
)

pip show opencv-python >nul 2>&1
if errorlevel 1 (
    pip install opencv-python
)

echo.
echo [2/3] 运行测试...
python scripts\auto_test_dmpr.py --images dataset\images --output test_results

echo.
echo [3/3] 打开报告...
start test_results\report.html

echo.
echo ========================================
echo 测试完成
echo ========================================
popd
pause
