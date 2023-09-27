# NVFLARE JOB TEMPLATE REGISTRY

This directory contains NVFLARE job templates. 

## Introduction

Each job template contains the following informations

* client-side configuration: config_fed_client.conf
* server-side configuration: config_fed_server.conf
* job meta info: meta.conf
* (Optional) data exchange configuration: config_exchange.conf. This is only used with the new FLARE ML to FL transition Client API
* information card: info.md for display purpose
* information config: used by program

# Configuration format

Configurations are written in HOCON (human optimized object Notation). As a variant of JSON, .conf can also use json format.
The pyhocon format allows for comments, and you can remove many of the double quotes as well as replace ":" with "=" to make the configurations look cleaner.
You can find details in [pyhoconb: HOCON Parser for python](https://github.com/chimpler/pyhocon). 
