pkill -9 python
bash utils/log_memory.sh >>/tmp/nvflare/workspace/container.txt &
python streaming_job.py --retriever_mode container
