## Federated Statistics

#### Client Executor need to do 
* specify the data path 
* define the data_load() function to load data
* define client_data_validate to make sure all data features are numerical features
* define specified local metrics needed to do the calculation, sum, count, var, stddev, histogram

#### Controller will 
* the job will calculate the global metrics for each feature

#### todo: 
* examples using Pandas data frame from CSV file
* examples with more than one dataset from one site (train/test)
* examples using TFRecords or some other dataformat
* Unit tests
* extract common libs if needed