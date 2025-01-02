docker run -d \
  --name statsd-prometheus-exporter \
  -p 9102:9102 \
  -e STATS_D_HOST=statsd \
  -e PROMETHEUS_PORT=9102 \
  statsd/statsd-prometheus-exporter
