#!/usr/bin/env bash
site_name=site-NNN
sp_end_point=localhost:8002:8003
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "WORKSPACE set to $DIR/.."
mkdir -p $DIR/../transfer
if [ $# -eq 0 ] ; then
	echo "No sever name is provided.  Default server (localhost) and sp_end_point (localhost:8002:8003) is used"
	echo "Usage: start.sh <SERVER_HOST <CLIENT_NAME <SP_END_POINT>>>"
	server=localhost
elif [ $# -eq 1 ] ; then
	server=$1
elif [ $# -eq 2 ] ; then
	server=$1
	site_name=$2
elif [ $# -eq 3 ] ; then
    server=$1
	site_name=$2
	sp_end_point=$3
else
	echo "Usage: start.sh <SERVER_HOST <CLIENT_NAME <SP_END_POINT>>>"
	exit
fi
sleep 1
$DIR/sub_start.sh $site_name $server $sp_end_point &
