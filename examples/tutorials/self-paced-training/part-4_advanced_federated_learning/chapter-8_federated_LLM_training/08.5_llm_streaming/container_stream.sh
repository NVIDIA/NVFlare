bash utils/log_memory.sh >>/tmp/nvflare/logs/container.txt &
python job.py --retriever_mode container
