mkdir /tmp/nvflare/logs/
bash utils/log_memory.sh >>/tmp/nvflare/logs/regular.txt &
python job.py
