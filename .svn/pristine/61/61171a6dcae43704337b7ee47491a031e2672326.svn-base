@echo off

if "%~1" == "" goto :usage
set backupdir=%~2
if "%backupdir%" == "" set backupdir=%~1

for %%F in ("%~f1\*.rsk") do call :process "%%F" "%backupdir%" 5
for %%F in ("%~f1\*.idm") do call :process "%%F" "%backupdir%" 8
exit /b

:process
if exist "%~f2\%~nx1.orig" goto :skip
echo Processing %~nx1
move "%~1" "%~f2\%~nx1.orig" > nul
cscript //Nologo RskConvert.js %3 < "%~f2\%~nx1.orig" > "%~1"
exit /b

:skip
rem echo "Orig file for %~1 already exists--skipping"
exit /b

:usage
echo Usage: %0 directory [backup directory]
exit /b
