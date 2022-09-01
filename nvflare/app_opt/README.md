 The nvflare/app_opt is the module with components or files which **optional** dependencies. 
 That means there are dependencies are not required to be installed when user type
 ```
     pip install nvflare
 ```

User will import errors when using the functions in this module but the required dependency is not installed.
In this module, instead of directly using 
```
  from torch.nn.functional import conv1d
```
  at top of the file, we encourage user to use option_import() function when you needed, for example

```
   from nvflare.utils.import_utils import optional_import
   ...
   < your code>
   torch = optional_import(module="torch.nn.functional", name = "conv1d")
   <rest of code>
      
```
   if the torch is not installed, use of torch (calling any attributes or function) will throw LazyImportError
with proper error messages. 
   