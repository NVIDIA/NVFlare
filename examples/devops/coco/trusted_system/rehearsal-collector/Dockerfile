FROM ubuntu@sha256:1e0a86e57d247923571b75e0aaf48a1449cf8c543d51fb3e07a4a7d7bfa79316

COPY snpguest /usr/local/bin/snpguest
COPY collect-snp-evidence.sh /usr/local/bin/collect-snp-evidence
RUN chmod 0755 /usr/local/bin/snpguest /usr/local/bin/collect-snp-evidence

ENTRYPOINT ["/usr/local/bin/collect-snp-evidence"]
