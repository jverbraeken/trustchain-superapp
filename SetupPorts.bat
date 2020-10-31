@echo off
set ports[0]=5554
set ports[1]=5556
set ports[2]=5558
set redirects[0]=65460
set redirects[1]=65451
set redirects[2]=59001

set "x=0"
:SymLoop
if defined ports[%x%] (
	call echo %%ports[%x%]%%
    set /a "x+=1"
    GOTO :SymLoop 
)
set /a "x-=1"
echo "Redirecting %x% AVDs"

setlocal EnableDelayedExpansion
for /L %%i in (0, 1, %x%) do (
	echo Forwarding AVD !ports[%%i]! to port !redirects[%%i]!
	start telnet.exe localhost !ports[%%i]!
	cscript TelnetHelper.vbs !redirects[%%i]!
)
endlocal