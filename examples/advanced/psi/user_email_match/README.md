# NVFlare PSI for user email matching

## How to use FLARE PSI operator

For User ids matching, the usage is really simple. 

* Step 1: user needs to implement the PSI interface where the client side's items need to be loaded.
These items could be user_ids or feature names depending on your use case.

```
class LocalPSI(PSI):
    def load_items(self) -> List[str]:
        pass

```

* Step 2: Use DhPSIRecipe

```
recipe = DhPSIRecipe(
    name="user_email_match",
    min_clients=args.n_clients,
    local_psi=local_psi,
    output_path=args.psi_output_path,
)
```


## Run PSI job

### User_email_match 
   in this example, we generated some random emails addresses in three sites. 
   you need to copy the data to tmp location first. 

**prepare data**

change to the PSI example directory
```
cd NVFlare/examples/advanced/psi/user_email_match
```
We have already prepared some random fake emails as data, all we need to copy this dat to a location 
that used by the data loading code, we have specified "/tmp/nvflare/psi" in our sample code, so we copy the data to
"/tmp/nvflare/psi" directory

```
./prepare_data.sh
```   
You will see something like the followings

```
copy NVFlare/examples/advanced/psi/user_email_match/data to /tmp/nvflare/psi directory

```

**import note**
   The items must be unique. duplicate items can result incorrect intersection result

**run job** 
```
python job.py --n_clients 3 --workspace_root /tmp/nvflare/psi/
```
Once job completed and succeed, you should be able to find the intersection for different sites at

```
/tmp/nvflare/psi/user_email_match/site-1/simulate_job/site-1/psi/intersection.txt
/tmp/nvflare/psi/user_email_match/site-2/simulate_job/site-2/psi/intersection.txt
/tmp/nvflare/psi/user_email_match/site-3/simulate_job/site-3/psi/intersection.txt
```
to compare these intersections, you can check with the followings:

```bash
diff <(sort /tmp/nvflare/psi/user_email_match/site-1/simulate_job/site-1/psi/intersection.txt) <(sort /tmp/nvflare/psi/user_email_match/site-2/simulate_job/site-2/psi/intersection.txt)
diff <(sort /tmp/nvflare/psi/user_email_match/site-2/simulate_job/site-2/psi/intersection.txt) <(sort /tmp/nvflare/psi/user_email_match/site-3/simulate_job/site-3/psi/intersection.txt)
```

**NOTE**
>>The PSI operator depends on openmind-psi. It now supports up-to-python 3.11
python 3.12 is still working in progress