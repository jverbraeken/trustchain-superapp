@echo off
setlocal
adb -s emulator-5554 root
adb -s emulator-5554 push ./NodeAssignment.csv /data/user/0/nl.tudelft.trustchain/files/
adb -s emulator-5556 root
adb -s emulator-5556 push ./NodeAssignment.csv /data/user/0/nl.tudelft.trustchain/files/
adb -s emulator-5558 root
adb -s emulator-5558 push ./NodeAssignment.csv /data/user/0/nl.tudelft.trustchain/files/
adb -s emulator-5560 root
adb -s emulator-5560 push ./NodeAssignment.csv /data/user/0/nl.tudelft.trustchain/files/
endlocal
pause