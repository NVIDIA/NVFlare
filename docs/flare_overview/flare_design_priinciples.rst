
Design Principles
=================

* Keep it simple, less is more
* design to specification
* gear to real-world scenarios
* keep system general-purpose
* client system friendly

**Less is more**

we like to solve unique challenges, we do less, but enable others tp dp more.
we can's solve whole worlds problem, but enable others to solve world's problem

This design principle means we intentioned limit the scope of the implementation,
but only implements the needed components. For given implementation we follow the specs and allows others to customize.


**Design to specification**

Every components, API design are spec-based, system can have alternative implementation following the spec.
This allows pretty much every component to be customized.

This also means that we are not opinionated with particular implementation, you are encourage to customize.


**gear to real-world scenarios**

handle real-world use cases where handle failure gracefully
default implementation will provide the implementation that can solve real-world problem


**Keep system general-purpose**

keep system general-purpose, enable different "federated" computing use cases.
We carefully package the components into different layers, the lower-layers has no dependency for upper layers
Specific use cases should not demands specific implementation of the underline system core.


**client system friendly**

make FLARE works well in your system
