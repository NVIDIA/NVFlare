# Optional dependency modules
Some modules have optional dependency such as
* nvflare.app_opt

The nvflare/app_opt module contains components with **optional** dependencies.
That means there are dependencies are not required to be installed when user type
 ```
     pip install nvflare
 ```
* to install optional dependency for application level module, use

```
  pip install nvflare[app_opt]
```
to install all optional dependency, use

```
  pip install nvflare[all]
```
