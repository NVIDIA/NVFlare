pkill -9 python
mkdir /tmp/nvflare/workspace/
bash utils/log_memory.sh >>/tmp/nvflare/workspace/regular.txt &
python streaming_job.py
