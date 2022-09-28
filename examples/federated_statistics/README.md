# Objective
Federated Statistics will provide built-in NVFLARE federated statistics operators ( controller and executors) that 
will generate global statistics based on local client side statistics.

At each clide site, we could have one or more datasets (such as "train" and "test" datasets); each dataset may have many 
features. For each feature in the dataset, we will calculate the statistics and combined to produce 
global statistics for all the numeric features. The output would be complete statistics for all datasets in clients and global.    

The statistics used here are commonly used statistics: count, sum, mean, std_dev and histogram for the numerical features. 

If the statistics sum and count are selected, the mean will be calculated with count and sum. 

A client will only need to implement the "Statistics" class from statistics_spec 

* configure the config_fed_server.json to indicate the specific statistics you need
  * Note: count is always required as we use count to enforce data privacy policy
* Client will required to supply features ( feature name, and data type) of datasets 
* Client will need to provides local statistics for given dataset and feature
* Categorical features are filter out, remains are numerical features
* 
* Output directory
  * The result will be saved to job workspace, which can be downloaded via download_job command 
  * The result is in json format, which can be loaded in pandas data frame. 
  * We provided some visualization utilities, one can visualize via jupyter notebook

# How it works

```mermaid
 
sequenceDiagram
    participant FileStore
    participant Server
    participant PrivacyFilter
    participant Client
    participant Statistics
    Server->>Client: pre_run: optional handshake (if enabled), passing targeted statistic configs
    Server->>Client: task: Fed_Stats: statistics_task_1: count, sum, mean,std_dev, min, max 
    loop over dataset and features
       Client->>Stats_Generator: local stats calculation
    end
    Client-->>PrivacyFilter: local statistic
     loop over statistics_privacy_filters
        PrivacyFilter->>PrivacyFilter: min_count_cleanser, min_max_cleanser, histogram_bins_cleanser
    end
    PrivacyFilter-->>Server: filtered local statistic
    loop over clients
        Server->>Server: aggregatation
    end
    Server->>Client:  task: Fed_Stats: statistics_task_2: var with input global_mean, global_count, histogram with estimated global min/max
    loop over dataset and features
       Client->>Stats_Generator: local stats calculation
    end
    Client-->>PrivacyFilter: local statistics: var, Histogram, count
    loop over statistics_privacy_filters
        PrivacyFilter->>PrivacyFilter: histogram_max_bins_check, min_count_check
    end
    PrivacyFilter-->>Server: filtered local statistic    
    loop over clients
        Server->>Server: aggregate var, std_dev, histogram
    end
    Server->>FileStore: save to file
```

```
