echo $PYTHONPATH

n_clients=$1

if test -z "${n_clients}"
then
      echo "Usage: ./stop_fl_secure.sh [n_clients], e.g. ./stop_fl_secure.sh 8"
      exit 1
fi

SERVERNAME="localhost"
workspace="secure_workspace"
site_pre="site-"

echo "Attempting to shutdown server at ${workspace}/${SERVERNAME}"
export CUDA_VISIBLE_DEVICES=1
echo 'y' | ./workspaces/${workspace}/${SERVERNAME}/startup/stop_fl.sh &

echo "Attempting to shutdown ${n_clients} clients"
for id in $(eval echo "{1..$n_clients}")
do
  export CUDA_VISIBLE_DEVICES=1
  echo 'y' | ./workspaces/${workspace}/${site_pre}${id}/startup/stop_fl.sh &
done
