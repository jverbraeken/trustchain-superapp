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
for /L %%x in (1, 1, %idi%) do (
adb -s !var%%x! root
adb -s !var%%x! install -t ../app/build/outputs/apk/debug/app-debug.apk
adb -s !var%%x! shell am start -n nl.tudelft.trustchain/nl.tudelft.trustchain.app.ui.dashboard.DashboardActivity -e activity fedml -e dataset mnist -e optimizer adam -e learningRate rate_1em3 -e momentum none -e l2Regularization l2_5em3 -e batchSize batch_32 -e epoch epoch_50 -e iteratorDistribution mnist_1 -e maxTestSample num_50 -e gar mozi -e communicationPattern random -e behavior noise -e runner distributed -e run false
)
endlocal
pause