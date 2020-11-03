@echo off
setlocal
adb -s emulator-5554 shell am force-stop nl.tudelft.trustchain
adb -s emulator-5556 shell am force-stop nl.tudelft.trustchain
adb -s emulator-5558 shell am force-stop nl.tudelft.trustchain
adb -s emulator-5560 shell am force-stop nl.tudelft.trustchain
endlocal
pause