docker run -d \
  --name statsd \
  -p 8125:8125/udp \
  -e STATSD_LISTENING_INTERFACE=0.0.0.0 \
  -e STATSD_PORT=8125 \
  -e STATSD_BACKEND=none \
  rockstat/statsd
