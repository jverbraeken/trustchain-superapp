@echo off
setlocal
adb -s emulator-5554 root
adb -s emulator-5554 install -t ../app/build/outputs/apk/debug/app-debug.apk
adb -s emulator-5554 shell am start -n nl.tudelft.trustchain/nl.tudelft.trustchain.app.ui.dashboard.DashboardActivity -e activity fedml -e dataset mnist -e optimizer adam -e learningRate rate_1em3 -e momentum none -e l2Regularization l2_5em3 -e batchSize batch_32 -e epoch epoch_50 -e iteratorDistribution mnist_1 -e maxTestSample num_50 -e gar mozi -e communicationPattern random -e behavior noise -e runner distributed -e run false

adb -s emulator-5556 root
adb -s emulator-5556 install -t ../app/build/outputs/apk/debug/app-debug.apk
adb -s emulator-5556 shell am start -n nl.tudelft.trustchain/nl.tudelft.trustchain.app.ui.dashboard.DashboardActivity -e activity fedml -e dataset mnist -e optimizer adam -e learningRate rate_1em3 -e momentum none -e l2Regularization l2_5em3 -e batchSize batch_32 -e epoch epoch_50 -e iteratorDistribution mnist_1 -e maxTestSample num_50 -e gar mozi -e communicationPattern random -e behavior benign -e runner distributed -e run false

adb -s emulator-5558 root
adb -s emulator-5558 install -t ../app/build/outputs/apk/debug/app-debug.apk
adb -s emulator-5558 shell am start -n nl.tudelft.trustchain/nl.tudelft.trustchain.app.ui.dashboard.DashboardActivity -e activity fedml -e dataset mnist -e optimizer adam -e learningRate rate_1em3 -e momentum none -e l2Regularization l2_5em3 -e batchSize batch_32 -e epoch epoch_50 -e iteratorDistribution mnist_1 -e maxTestSample num_50 -e gar mozi -e communicationPattern random -e behavior benign -e runner distributed -e run false

adb -s emulator-5560 root
adb -s emulator-5560 install -t ../app/build/outputs/apk/debug/app-debug.apk
adb -s emulator-5560 shell am start -n nl.tudelft.trustchain/nl.tudelft.trustchain.app.ui.dashboard.DashboardActivity -e activity fedml -e dataset mnist -e optimizer adam -e learningRate rate_1em3 -e momentum none -e l2Regularization l2_5em3 -e batchSize batch_32 -e epoch epoch_50 -e iteratorDistribution mnist_1 -e maxTestSample num_50 -e gar mozi -e communicationPattern random -e behavior benign -e runner distributed -e run false
endlocal
pause