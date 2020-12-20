set OBJECT=WScript.CreateObject("WScript.Shell")
WScript.sleep 50
OBJECT.SendKeys "auth fqvo8zH1j32aFoVB{ENTER}"
WScript.sleep 50
OBJECT.SendKeys "redir add udp:" & WScript.Arguments.Item(0) & ":8090{ENTER}"
WScript.sleep 50
OBJECT.SendKeys "exit{ENTER}" 