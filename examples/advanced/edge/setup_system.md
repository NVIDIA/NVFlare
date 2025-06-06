# Running Edge Example with Hierarchical Clients

Please follow these steps to run the DeviceSimulator,

## Provision

Use tree_prov.py to generate a hierarchical NVFlare system with 2 levels and 2 clients at each level,

     python nvflare/edge/tree_prov.py -r /tmp -p edge_example -d 1 -w 2

This will create a deployment with 4 leaf nodes, 2 aggregators, 2 relays, and 1 server.

To start the system, just run the following command in the prepared workspace,

```commandline
cd /tmp/edge_example/prod_00
./start_all.sh
```    

## Starting Web Proxy

To route devices to different LCP, routing_proxy is used. It's a simple proxy that routes the request to
different LCP based on checksum of the device ID. It can be started like this,

    python nvflare/edge/web/routing_proxy.py 8000 /tmp/edge_example/lcp_map.json

The lcp_map.json file is generated by tree_prov.py.

## Example Job

The `hello_mobile` is a very simple job to test the edge functions. It only sends one task "train"  and
print the aggregated results.

The job can be started as usual in NVFlare admin console.

## Run DeviceSimulator

The DeviceSimulator can be used to test all the features of the edge system. It handles 'train' task by simply doubling every values 
in the weights.

To start the DeviceSimulator, give it an endpoint URL and number of devices like this,

    python /nvflare/edge/device_simulator/run_device_simulator.py http://localhost:8000 16
   
The DeviceSimulator keeps polling the LCP for job assignment. It only runs one job then quits.



