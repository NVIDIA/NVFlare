# Federated Private Set Intersection

The Private Set Intersection (PSI) protocol is based on [ECDH](https://en.wikipedia.org/wiki/Elliptic-curve_Diffie%E2%80%93Hellman),
Bloom Filters, and Golomb Compressed Sets PSI algorithm. The algorithm is developed by [openmined PSI](https://github.com/OpenMined/PSI)
for two-party. 

We took the two-party direct communication PSI protocol and extended to Federated Computing setting where all exchanges are 
funneled via a central FL server. We supported multi-party PSI via pair-wise approach.  
 
Here is the detailed Sequence diagrams. 

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
    PSIController->>PSIWorkflow: pre_workflow() --> prepare_sites()
    loop over sites:
        PSIWorkflow -->> site_1 : psi prepare
        site_1 -->> PSI : load_items() 
        PSI -->> site_1: item size
        site_1 -->> PSIWorkflow: receive site 1 items size
    end

    loop over sites:
        PSIWorkflow -->> PSIWorkflow : sort the site according to size 
    end 
    
    PSIController->>PSIWorkflow: workflow() --> setup, request, response, intersects
    note over PSIWorkflow, PSI : Forward Pass
    loop over sites: 
        
        PSIWorkflow -->> site_1 : setup task, arg: site-2 item size
        site_1 -->> site_1 :  load items() if intersect: load intersections else load_items() 
        site_1 -->> PSIWorkflow: receive for reach site setup message for site-2
            
        PSIWorkflow -->> site_2 : send site-1 setup_msg and ask for request 
        site_2 -->> site_2 : save setup msg
        site_2 -->> site_2 : load items() if intersect: load intersections else load_items() 
        site_2 -->> site_2 : cretea request message
         
        site_2 -->> PSIWorkflow: reciev request message  
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
* Once the client has get the intersection from the previous Clients' intersect operation, the items will be
* the intersection instead of original items.

```mermaid
 
sequenceDiagram

    participant FLServer
    participant PSIExecutor
    participant DhPSIExecutor
    participant PsiServer
    participant PsiClient
    participant PSI
    participant FilePsiWriter
    
    Note over FLServer, PSI : PREPARE 
    FLServer -->>PSIExecutor: initialize()
    PSIExecutor -->> PSIExecutor: get_client_executor()
    PSIExecutor -->> PSIExecutor: psi_client_executor = DhPSIExecutor(local_psi_id)
    FLServer -->> PSIExecutor: execute()
        
    Note over FLServer, PSI : PREPRAE
    PSIExecutor -->> DhPSIExecutor: client_exec() : PSIConst.PSI_TASK_PREPARE
    DhPSIExecutor -->> DhPSIExecutor : PSI Prepare with fpr
    DhPSIExecutor -->> PSI : load_terms()
    PSI -->> DhPSIExecutor: items
    DhPSIExecutor -->> DhPSIExecutor: load PsiClient(items)
    DhPSIExecutor -->> DhPSIExecutor: load PsiServer(items, fpr)
    DhPSIExecutor -->> PSIExecutor: result
    PSIExecutor -->> FLServer : items size
    
    Note over FLServer, PSI : SETUP
    FLServer -->> PSIExecutor : PSI Setup
    PSIExecutor -->> DhPSIExecutor:   PSI Setup
    DhPSIExecutor -->> DhPSIExecutor: setup(client_items_size)
    DhPSIExecutor -->> DhPSIExecutor : get_items(): items = intersection or PSI.load_items() 
    DhPSIExecutor -->> DhPSIExecutor: load PsiClient(items)
    DhPSIExecutor -->> DhPSIExecutor: load PsiServer(items, fpr)
    DhPSIExecutor -->> PsiServer: setup(client_items_size)
    PsiServer -->> DhPSIExecutor: setup_msg
    DhPSIExecutor -->> PSIExecutor: result
    PSIExecutor -->> FLServer: setup_msg
    
    Note over FLServer, PSI : create Request
    FLServer -->> PSIExecutor :  PSI create_request, with setup message
    PSIExecutor -->> DhPSIExecutor:  PSI create_request, with setup message
    DhPSIExecutor -->> PsiClient: save (setup_msg), 
    DhPSIExecutor -->> DhPSIExecutor : items = get_items() : intersection or PSI.load_items()
    DhPSIExecutor -->> PsiClient : get_request(items )
    PsiClient -->> DhPSIExecutor : request_msg
    DhPSIExecutor -->> PSIExecutor: result
    PSIExecutor -->> FLServer:request_msg
 
    Note over FLServer, PSI : process Request
    FLServer -->> PSIExecutor : PSI process_request, with request_msg
    PSIExecutor -->> DhPSIExecutor : PSI process_request, with request_msg
    DhPSIExecutor -->> PsiServer: process_request (request_msg), 
    PsiServer -->> DhPSIExecutor : response
    DhPSIExecutor -->> PSIExecutor: result
    PSIExecutor -->> FLServer : response
 
    Note over FLServer, PSI : calculate intersection
    FLServer -->> PSIExecutor : calculate_intersect with response msg
    PSIExecutor -->> DhPSIExecutor : calculate_intersect with response msg
    DhPSIExecutor -->> PsiClient: get_intersection (response_msg) 
    PsiClient -->> DhPSIExecutor : intersection
    DhPSIExecutor -->> PSI : save intersection
    PSI -->> FilePsiWriter : save intersection
    DhPSIExecutor -->> PSIExecutor: result
    PSIExecutor -->> FLServer : status
    
```



