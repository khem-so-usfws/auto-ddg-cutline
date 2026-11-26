@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0" || exit /b 1
python -m tools.build_windows %*
set "build_status=%ERRORLEVEL%"
popd
exit /b %build_status%
