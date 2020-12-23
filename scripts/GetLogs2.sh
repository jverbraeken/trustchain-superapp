#!/bin/sh
DEVICES=`adb devices | grep -v devices | grep device | cut -f 1`
for device in $DEVICES; do
	echo "$device $@ ..."
	echo "adb -s $device pull /data/user/0/nl.tudelft.trustchain/files/evaluations/ $HOME/Downloads/evaluations/$device"
	adb -s $device root
	adb -s $device pull /data/user/0/nl.tudelft.trustchain/files/evaluations/ $HOME/Downloads/evaluations/$device
done
