set OBJECT=WScript.CreateObject("WScript.Shell")
WScript.sleep 25
OBJECT.SendKeys "auth fqvo8zH1j32aFoVB{ENTER}"
WScript.sleep 25
OBJECT.SendKeys "redir add udp:" & WScript.Arguments.Item(0) & ":8090{ENTER}"
WScript.sleep 25
'WScript.sleep 1500
'OBJECT.SendKeys "exit{ENTER}" 