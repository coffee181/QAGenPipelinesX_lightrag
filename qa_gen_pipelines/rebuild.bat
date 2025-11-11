@echo off
echo 🔧 QA生成管道快速重新打包
echo ====================================

echo 📁 当前目录: %CD%

echo 🧹 清理旧的构建文件...
if exist "build" rmdir /s /q "build" 2>nul
if exist "dist" rmdir /s /q "dist" 2>nul  
if exist "build_venv" rmdir /s /q "build_venv" 2>nul
if exist "deployment" rmdir /s /q "deployment" 2>nul
del *.spec 2>nul

echo 🚀 开始重新打包...
python build_with_venv.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 打包完成！
    echo 📦 可执行文件位于: deployment\qa_gen_pipeline.exe
    echo 📄 使用说明: deployment\README.md
    echo.
    echo 💡 快速测试:
    echo    cd deployment
    echo    qa_gen_pipeline.exe --help
    echo.
) else (
    echo.
    echo ❌ 打包失败！
    echo 请检查错误信息并重试
    echo.
)

pause 