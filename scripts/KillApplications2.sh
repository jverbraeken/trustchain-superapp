#!/bin/sh
DEVICES=`adb devices | grep -v devices | grep device | cut -f 1`
for device in $DEVICES; do
	adb -s $device shell am force-stop nl.tudelft.trustchain
done
