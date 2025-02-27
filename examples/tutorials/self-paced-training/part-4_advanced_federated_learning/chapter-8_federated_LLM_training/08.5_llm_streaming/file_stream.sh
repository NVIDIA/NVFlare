bash utils/log_memory.sh >>/tmp/nvflare/logs/file.txt &
python streaming_job.py --retriever_mode file
