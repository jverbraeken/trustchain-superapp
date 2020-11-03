@echo off
setlocal enableDelayedExpansion
set "replace=device"
set "replaced="
set "source=tmp2"
set "target=tmp3"
adb devices > tmp1
findstr /v "List of devices attached" tmp1 > tmp2 (
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
	adb -s !var%%x! shell am force-stop nl.tudelft.trustchain
)
endlocal
pause