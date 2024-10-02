# NVFlare PSI for user email matching

## How to use FLARE PSI operator ?

For User ids matching, the usage is really simple. 

* Step 1: user needs to implement the PSI interface where the client side's items need to be loaded.
These items could be user_ids or feature names depending on your use case.

* Step 2: Specify the job configurations

**Client Config**

```
{
  format_version = 2
  executors = [
    {
      tasks = ["PSI"]
      executor {
        id = "Executor"
        path = "nvflare.app_common.psi.psi_executor.PSIExecutor"
        args.psi_algo_id = "dh_psi"
      }
    }
  ]

  components = [
    {
      id = "dh_psi"
      path = "nvflare.app_opt.psi.dh_psi.dh_psi_task_handler.DhPSITaskHandler"
      args.local_psi_id = "local_psi"
    },
    {
      id = "local_psi"
      path = "local_psi.LocalPSI"
      args {
        psi_writer_id = "psi_writer"
        data_root_dir = "/tmp/nvflare/psi/data"
      }
    },
    {
      id = "psi_writer",
      path = "nvflare.app_common.psi.file_psi_writer.FilePSIWriter"
      args.output_path = "psi/intersection.txt"
    }
  ]
}
```

Here we specify the following components:

* **_PSIExecutor_** : this is the built-in FLARE PSIExecutor. 
* **_local_psi_** : local PSI component is the component user needs to write, here we called it "local_psi". 
the local psi component require PSIWriter, so can one save the resulting intersect to storage. here we implemented
a file writer
* **_FilePSIWriter_** : save the intersection to a file in workspace.  

**Server Config**

Just specify the built-in PSI controller. 
```
{
  format_version = 2,
  workflows = [
    {
      id = "DhPSIController"
      path = "nvflare.app_common.psi.dh_psi.dh_psi_controller.DhPSIController"
      args{
      }
    }
  ]
}
```
**Code**
 the code is really trivial just needs to implement one method in PSI interface

```
class LocalPSI(PSI):
    def load_items(self) -> List[str]:
        pass

```

## Run PSI job in Simulator

### User_email_match 
   in this example, we generated some random emails addresses in three sites. 
   you need to copy the data to tmp location first. 

**prepare data**

change to the PSI example directory
```
cd NVFlare/examples/advanced/psi/
```
We have already prepared some random fake emails as data, all we need to copy this dat to a location 
that used by the data loading code, we have specified "/tmp/nvflare/psi" in our sample code, so we copy the data to
"/tmp/nvflare/psi" directory

```
user_email_match/prepare_data.sh
```   
You will see something like the followings

```
copy NVFlare/examples/advanced/psi/user_email_match/data to /tmp/nvflare/psi directory

```

**import note**
   The items must be unique. duplicate items can result incorrect intersection result

**run job** 
```
nvflare simulator -w /tmp/nvflare/psi/job -n 3 -t 3 user_email_match/jobs/user_email_match
```
Once job completed and succeed, you should be able to find the intersection for different sites at

```
/tmp/nvflare/psi/job/simulate_job/site-1/psi/intersection.txt 
/tmp/nvflare/psi/job/simulate_job/site-2/psi/intersection.txt 
/tmp/nvflare/psi/job/simulate_job/site-3/psi/intersection.txt  
```
to compare these intersections, you can check with the followings:

```bash
diff <(sort /tmp/nvflare/psi/job/simulate_job/site-1/psi/intersection.txt) <(sort /tmp/nvflare/psi/job/simulate_job/site-2/psi/intersection.txt)
diff <(sort /tmp/nvflare/psi/job/simulate_job/site-2/psi/intersection.txt) <(sort /tmp/nvflare/psi/job/simulate_job/site-3/psi/intersection.txt)
```

**NOTE**
>>The PSI operator depends on openmind-psi. It now supports up-to-python 3.11
python 3.12 is still working in progress