#!/usr/bin/env bash
site_name=site-NNN
sp_end_point=localhost:8002:8003
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "WORKSPACE set to $DIR/.."
mkdir -p $DIR/../transfer
if [ $# -eq 0 ] ; then
	echo "No sp_end_point is provided.  Default sp_end_point (localhost:8002:8003) is used"
	echo "Usage: start.sh <SP_END_POINT <CLIENT_NAME>>"
	server=localhost
elif [ $# -eq 1 ] ; then
	sp_end_point=$1
elif [ $# -eq 2 ] ; then
	sp_end_point=$1
	site_name=$2
else
	echo "Usage: start.sh <SP_END_POINT <CLIENT_NAME>>"
	exit
fi
sleep 1
$DIR/sub_start.sh $site_name $sp_end_point &
