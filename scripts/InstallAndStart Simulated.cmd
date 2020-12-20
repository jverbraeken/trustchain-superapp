@echo off
setlocal enableDelayedExpansion
set "replace=device"
set "replaced="
set "source=tmp2"
set "target=tmp3"
adb devices > tmp1
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set datetime=%datetime:~0,8%-%datetime:~8,6%
findstr /v "List of devices attached" tmp1 > tmp2
(
   for /F "tokens=1* delims=:" %%a in ('findstr /N "^" %source%') do (
      set "line=%%b"
      if defined line set "line=!line:%replace%=%replaced%!"
      echo(!line!)
) > %target%
set idi=0
for /F "usebackq" %%A in (tmp3) do (
    SET /A idi=!idi! + 1
    set var!idi!=%%A
)
set var
adb -s emulator-5554 root
adb -s emulator-5556 root
adb -s emulator-5554 install -t ../app/build/outputs/apk/debug/app-debug.apk
adb -s emulator-5556 install -t ../app/build/outputs/apk/debug/app-debug.apk
adb -s emulator-5554 shell am start -n nl.tudelft.trustchain/nl.tudelft.trustchain.app.ui.dashboard.DashboardActivity -e activity fedml -e automationPart 1
adb -s emulator-5556 shell am start -n nl.tudelft.trustchain/nl.tudelft.trustchain.app.ui.dashboard.DashboardActivity -e activity fedml -e automationPart 2
endlocal
pause