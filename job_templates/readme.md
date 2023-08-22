# NVFLARE JOB TEMPLATE REGISTRY

This directory contains NVFLARE job templates. 

## Introduction

Each job template contains the following informations

* client-side configuration: config_fed_client.conf
* server-side configuration: config_fed_server.conf
* job meta info  meta.conf
* (Optionally) data exchange configuration: config_exchange.conf . This is only used with the newly FLARE ML to FL transition Client API
* information card: info.md for display purpose
* information config: used by program

# Configuration format

  The configuration is written as HOCON (human optimized object Notation), it is an variant of JSON,
  so variant of JSON, so the json format is ok. But you remove the many of the double quotes 
  replace ":" with "=". You can find the details in [pyhoconb: HOCON Parser for python](https://github.com/chimpler/pyhocon)

  With pyhocon format, we can add comments in each line. 

    
