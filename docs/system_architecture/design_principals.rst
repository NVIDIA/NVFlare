
Design Principles
=================

- Less is more
- Design to specification
- Build for real-world scenarios
- Keep the system general-purpose
- Client system friendly

**Less is more**
We strive to solve unique challenges by doing less while enabling others to do more.
We can't solve whole world's problems, but by building an open platform, we can enable others to solve them.
This design principle means we intentionally limit the scope of the implementation, only building the necessary components.
For a given implementation, we follow specifications in a way that allows others to easily customize and extend.

**Design to Specification**
Every component and API is specification-based, so that alternative implementations can be constructed by following the spec.
This allows pretty much every component to be customized.
We strive to be open-minded in reference implementations, encouraging developers and end-users to extend and customize to meet the needs of their specific workflows.

**Built for real-world scenarios**
We build to handle real-world use cases where unexpected events or misbehaving code can be handled in a way that allows components or the system as a whole to fail gracefully.
The reference implementations of the default components are designed to solve real-world problems in a straightforward way.

**Keep the system general-purpose**
We design the system to be general purpose, to enable different “federated” computing use cases.
We carefully package the components into different layers with minimal dependencies between layers.
In this way, implementations for specific use cases should not demand modifications to the underlying system core.

**Client system friendly**
We design the system so that it can run anywhere with minimal environmental dependencies.
We also strive to build the system in a way that does not interfere with the deployment environment, allowing FLARE to be easily integrated into your own applications or platforms.
