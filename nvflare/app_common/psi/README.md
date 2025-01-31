# Federated Private Set Intersection

The Private Set Intersection (PSI) protocol is based on [ECDH](https://en.wikipedia.org/wiki/Elliptic-curve_Diffie%E2%80%93Hellman),
Bloom Filters, and Golomb Compressed Sets PSI algorithm. The algorithm is developed by [openmined PSI](https://github.com/OpenMined/PSI)
for two-party. 

We took the two-party direct communication PSI protocol and extended to Federated Computing setting where all exchanges are 
funneled via a central FL server. We supported multi-party PSI via pair-wise approach.  
 
Here is the detailed sequence diagram for DH PSI.

```mermaid
 
sequenceDiagram
    
    participant FLServer
    participant PSIController 
    participant PSIWorkflow
    participant site_1
    participant site_2
    participant site_3
    participant PSI
    
    FLServer -->> PSIController: start_controller(): load PSIWorkflow, in our case DhPSIWorkflow 
    FLServer -->> PSIController: control_flow() 
    
    note over PSIWorkflow, PSI: Prepare Sites
    PSIController->>PSIWorkflow: pre_process() --> prepare_sites()
    loop over sites:
        PSIWorkflow -->> site_1 : psi prepare
        site_1 -->> PSI : load_items() 
        PSI -->> site_1: item size
        site_1 -->> PSIWorkflow: receive site 1 items size
    end

    loop over sites:
        PSIWorkflow -->> PSIWorkflow : sort the site according to size 
    end 
    
    PSIController->>PSIWorkflow: run() --> setup, request, response, intersects
    note over PSIWorkflow, PSI : Forward Pass
    loop over sites: 
        
        PSIWorkflow -->> site_1 : setup task, arg: site-2 item size
        site_1 -->> site_1 :  load items() if intersect: load intersections else load_items() 
        site_1 -->> PSIWorkflow: receive for reach site setup message for site-2
            
        PSIWorkflow -->> site_2 : send site-1 setup_msg and ask for request 
        site_2 -->> site_2 : save setup msg
        site_2 -->> site_2 : load items() if intersect: load intersections else load_items() 
        site_2 -->> site_2 : create request message
         
        site_2 -->> PSIWorkflow: receive request message  
        PSIWorkflow -->> site_1: send request msg from site-2 and ask for response
        site_1 -->> PSIWorkflow: send response
        PSIWorkflow -->> site_2: send response and ask for calculate intersection and save it
    end    
    
      note over PSIWorkflow, PSI : backward Pass: Sequential
    loop over sites:   
      PSIWorkflow -->> site_3 : setup, request, response, calculate intersection
      site_3 -->> PSIWorkflow : reverse the forward process
    end
    
    FLServer -->> PSIController: stop_controller()
    PSIController --> PSIWorkflow: finalize()
    

```
## Client Side interactions

* Note each site/client is both a PSI Client and PSI Server. 
* Initially, the items() is the original data items
* Once the client has gotten the intersection from the previous clients' intersect operation, the items will be
* the intersection instead of original items.

```mermaid
 
sequenceDiagram

    participant FLServer
    participant PSIExecutor
    participant DhPSITaskHandler
    participant PSIServer
    participant PSIClient
    participant PSI
    participant PSIWriter
    
    Note over FLServer, PSI : PREPARE 
    FLServer -->> PSIExecutor: initialize()
    PSIExecutor -->> PSIExecutor: get_task_handler()
    PSIExecutor -->> PSIExecutor: psi_task_handler = DhPSITaskHandler(local_psi_id)
    FLServer -->> PSIExecutor: execute()
        
    Note over FLServer, PSI : PREPARE
    PSIExecutor -->> DhPSITaskHandler: process() : PSIConst.PSI_TASK_PREPARE
    DhPSITaskHandler -->> DhPSITaskHandler : PSI Prepare with fpr
    DhPSITaskHandler -->> PSI : load_terms()
    PSI -->> DhPSITaskHandler: items
    DhPSITaskHandler -->> DhPSITaskHandler: load PSIClient(items)
    DhPSITaskHandler -->> DhPSITaskHandler: load PSIServer(items, fpr)
    DhPSITaskHandler -->> PSIExecutor: result
    PSIExecutor -->> FLServer : items size
    
    Note over FLServer, PSI : SETUP
    FLServer -->> PSIExecutor : PSI Setup
    PSIExecutor -->> DhPSITaskHandler:   PSI Setup
    DhPSITaskHandler -->> DhPSITaskHandler: setup(client_items_size)
    DhPSITaskHandler -->> DhPSITaskHandler : get_items(): items = intersection or PSI.load_items() 
    DhPSITaskHandler -->> DhPSITaskHandler: load PSIClient(items)
    DhPSITaskHandler -->> DhPSITaskHandler: load PSIServer(items, fpr)
    DhPSITaskHandler -->> PSIServer: setup(client_items_size)
    PSIServer -->> DhPSITaskHandler: setup_msg
    DhPSITaskHandler -->> PSIExecutor: result
    PSIExecutor -->> FLServer: setup_msg
    
    Note over FLServer, PSI : create Request
    FLServer -->> PSIExecutor :  PSI create_request, with setup message
    PSIExecutor -->> DhPSITaskHandler:  PSI create_request, with setup message
    DhPSITaskHandler -->> PSIClient: save (setup_msg), 
    DhPSITaskHandler -->> DhPSITaskHandler : items = get_items() : intersection or PSI.load_items()
    DhPSITaskHandler -->> PSIClient : get_request(items )
    PSIClient -->> DhPSITaskHandler : request_msg
    DhPSITaskHandler -->> PSIExecutor: result
    PSIExecutor -->> FLServer:request_msg
 
    Note over FLServer, PSI : process Request
    FLServer -->> PSIExecutor : PSI process_request, with request_msg
    PSIExecutor -->> DhPSITaskHandler : PSI process_request, with request_msg
    DhPSITaskHandler -->> PSIServer: process_request (request_msg), 
    PSIServer -->> DhPSITaskHandler : response
    DhPSITaskHandler -->> PSIExecutor: result
    PSIExecutor -->> FLServer : response
 
    Note over FLServer, PSI : calculate intersection
    FLServer -->> PSIExecutor : calculate_intersect with response msg
    PSIExecutor -->> DhPSITaskHandler : calculate_intersect with response msg
    DhPSITaskHandler -->> PSIClient: get_intersection (response_msg) 
    PSIClient -->> DhPSITaskHandler : intersection
    DhPSITaskHandler -->> PSI : save intersection
    PSI -->> PSIWriter : save intersection
    DhPSITaskHandler -->> PSIExecutor: result
    PSIExecutor -->> FLServer : status
    
```



