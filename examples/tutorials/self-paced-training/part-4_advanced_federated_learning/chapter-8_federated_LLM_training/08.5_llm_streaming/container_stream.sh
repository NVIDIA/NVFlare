bash utils/log_memory.sh >>/tmp/nvflare/logs/container.txt &
python streaming_job.py --retriever_mode container
