@echo off
set ports[0]=5554
set ports[1]=5556
set ports[2]=5558
set ports[3]=5560
set redirects[0]=64154
set redirects[1]=59625
set redirects[2]=58900
set redirects[3]=60274

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
	cscript SetupPortsHelper.vbs !redirects[%%i]!
)
endlocal
exit