
# local privacy policy

privacy.json provides local site specific privacy policy.
The policy is likely setup by the company and implemented by organization admin
for the project.

for different type of scope or categories, there are might be type of policy
in this example, illustrated statistics privacy policies via some built-in privacy filters. 

without the privacy.json in place, the federated statistics will have no privacy preservation. 

One need to manually copied the privacy configuration to the sites' privacy.json, assuming 
there are other privacy configurations for weights, weight diff, and many others.

# Privacy configuration

The NVFLARE privacy configuration is consists of set of task data filters and task result filters
* The task data filter applies before client executor executes;
* The task results filter applies after client executor before it sends to server;
* for both data filter and result filter, they are groups via scope.

Each job will need to have privacy scope. If not specified, the default scope will be used. If default scope is not 
defined and job doesn't specify the privacy scope, the job deployment will fail, and job will not executed

# Statistics Privacy Filters

Statistics privacy filters are task result filters.
```
StatisticsPrivacyFilter
```
The StatisticsPrivacyFilter is consists of three `StatisticsPrivacyCleanser`s focused on the statistics sent
from client to server. 

`StatisticsPrivacyCleanser` can be considered as an interceptor before the results delivered to server. 
Currently, we use three `StatisticsPrivacyCleanser`s to guard the data privacy

## MinCountCleanser:
check against the number of count returned from client for each dataset and each feature.

if the min_count is not satisfied, there is potential risk of reveal client's real data. Then remove that feature's statistics 
from the result for this client. 

## HistogramBinsCleanser: 
For histogram calculations, number of bins can't be too large compare to count. if the bins = count, then 
we also reveal the real data. This check to make sure that the number of bins be less than X percent of the count. 
X = max_bins_percent in percentage, for 10 is for 10%
if the number of bins for the histogram is not satisfy this specified condition, the resulting histogram will be removed 
from statistics before sending to server. 

## AddNoiseToMinMax
For histogram calculations, if the feature's histogram bin's range is not specified, we will need to use local data's min 
and max values to calculate the global min/max values, then use the global min, max values as the bin ragen for histogram 
calculation. But send the server the local min, max values will reveal client's real data.
To protect data privacy, we add noise to the local min/max values. 

min/max random is used to generate random noise between (min_noise_level and max_noise_level).
for example, the random noise is to be within (0.1 and 0.3),i.e. 10% to 30% level. These noise
will make local min values smaller than the true local min values, and max values larger than
the true local max values. As result, the estimate global max and min values (i.e. with noise)
are still bound the true global min/max values, in such that
```
est. global min value <
    true global min value <
        client's min value <
            client's max value <
                true global max <
                        est. global max value
```














