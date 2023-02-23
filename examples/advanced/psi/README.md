# Federated Private Set Intersection

In this example, we will demonstrate the built-in operator for Private Set Intersection (PSI).
What is PSI ? why do we need it and how to use it ?  

## What is PSI?

According to [Wikipedia](https://en.wikipedia.org/wiki/Private_set_intersection): ```The Private set intersection is a
secure multiparty computation cryptographic technique that allows two parties holding sets to compare encrypted versions 
of these sets in order to compute the intersection. In this scenario, neither party reveals anything to the counterparty
except for the elements in the intersection.```

![psi.png](psi.png)

## What's the use cases for PSI?

There are many use cases for PSI, in terms of federated machine learning, we are particularly interested in the 
following use cases:
* Vertical Learning -- User ids matching
  ![user_id_match.png](user_id_intersect.png)

* Vertical Learning -- feature overlapping discovery
  Site-1 : Feature A, B, C, D
  Site-2: Feature E, A, C, F, X, Y, Z
  Overlapping features: A, C

* Federated Statistics -- categorical feature distinct values count
  feature = email address
  discover :  how many distinct emails in the email addresses
  feature = country
  discover: how many distinct countries

  site-1:   features: country.  total distinct country = 20
  site-2:   features: country,  total distinct country = 100
  site-1 and site2 overlapping distinct country = 10  
  Total distinct countries = 20 + 100 - Overlapping countries  = 120-10 = 110

## PSI Protocol

There are many protocols that can be used for PSI, For this implementation, the Private Set Intersection (PSI) protocol is based on [ECDH](https://en.wikipedia.org/wiki/Elliptic-curve_Diffie%E2%80%93Hellman),
Bloom Filters, and Golomb Compressed Sets PSI algorithm. The algorithm is developed by [openmined PSI](https://github.com/OpenMined/PSI)
for two-party PSI.

We took the two-party direct communication PSI protocol and extended to Federated Computing setting where all exchanges are
funneled via a central FL server. We supported multi-party PSI via pair-wise approach.

## How to use FLARE PSI operator ? 

The usage is really simple. 

* Step 1: user needs to implement the PSI interface where the client side's items need to be loaded.
These items could be user_ids or feature names depending on your use case.

* Step 2: Specify the job configurations

**Client Config**
```
{
  "format_version": 2,
  "executors": [
    {
      "tasks": [
        "PSI"
      ],
      "executor": {
        "id": "Executor",
        "path": "nvflare.app_opt.psi.psi_executor.PSIExecutor",
        "args": {
          "local_psi_id": "local_psi"
        }
      }
    }
  ],
  "components": [
    {
      "id": "local_psi",
      "path": "local_psi.LocalPSI",
      "args": {
        "psi_writer_id": "psi_writer"
      }
    },
    {
      "id": "psi_writer",
      "path": "nvflare.app_common.psi.psi_file_writer.FilePsiWriter",
      "args": {
        "output_path": "psi/intersection.txt"
      }
    }
  ]
}

```
Here we specify the following components:

* **_PSIExecutor_** : this is built FLARE PSI Executor. 
* **_local_psi_** : local PSI component is the component user needs to write, here we called it "sample_psi". 
the local psi component require PSI Persistor, so can one save the resulting intersect to storage. here we implemented
a file writer
* **_FilePsiWriter_** : save the intersection to a file in workspace.  

**Server Config**

Just specify the built-in PSI controller. 
```
{
  "format_version": 2,
  "workflows": [
    {
      "id": "DhPSIController",
      "path": "nvflare.app_common.workflows.dh_psi_controller.DhPSIController",
      "args": {
      }
    }
  ]
}

```
**Code**
 the code is really trivial just needs to implement one method in PSI interface

```
class LocalPSI(PSI):

    def __init__(self, psi_writer_id: str):
        pass
    def load_items(self) -> List[str]:
        pass

```

## Run PSI job in Simulator

### User_email_match 
   in this example, we generated some random emails addresses in three sites. 
   you need to copy the data to tmp location first. 

**prepare data**
```
mkdir -p /tmp/nvflare/psi     
cp -r user_email_match/data /tmp/nvflare/psi/.
```   
**import note**
   The items must be unique. duplicate items can result incorrect intersection result

**run job** 
```
nvflare simulator -w /tmp/nvflare/psi -n 3 -t 3 examples/advanced/psi/user_email_match
```
Once job completed and succeed, you should be able to find the intersection for different sites at

```
/tmp/nvflare/psi/simulate_job/site-1/psi/intersection.txt 
/tmp/nvflare/psi/simulate_job/site-2/psi/intersection.txt 
/tmp/nvflare/psi/simulate_job/site-3/psi/intersection.txt  
```