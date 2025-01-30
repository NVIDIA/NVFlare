pkill -9 python
bash utils/log_memory.sh >>/tmp/nvflare/workspace/file.txt &
python streaming_job.py --retriever_mode file
