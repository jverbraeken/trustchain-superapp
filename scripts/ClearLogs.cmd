@echo off
setlocal
adb -s emulator-5554 logcat -b all -c
adb -s emulator-5556 logcat -b all -c
adb -s emulator-5558 logcat -b all -c
endlocal