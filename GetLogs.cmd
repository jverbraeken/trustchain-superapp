@echo off
setlocal
adb -s emulator-5554 root
adb -s emulator-5554 pull /data/user/0/nl.tudelft.trustchain/files/evaluations/ %USERPROFILE%\Downloads\evaluations\5554
adb -s emulator-5556 root
adb -s emulator-5556 pull /data/user/0/nl.tudelft.trustchain/files/evaluations/ %USERPROFILE%\Downloads\evaluations\5556
adb -s emulator-5558 root
adb -s emulator-5558 pull /data/user/0/nl.tudelft.trustchain/files/evaluations/ %USERPROFILE%\Downloads\evaluations\5558
adb -s emulator-5560 root
adb -s emulator-5560 pull /data/user/0/nl.tudelft.trustchain/files/evaluations/ %USERPROFILE%\Downloads\evaluations\5560
endlocal