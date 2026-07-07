This repository implements several workflows for automated experiments at a synchrotron beamline. The package depends on EAA (for source, see `<PROJECT_ROOT>/../eaa`) and should use the data structure (mainly `BaseTaskManager` and `BaseTool`) of EAA.

Refer to EAA's source code or documentation if you need to learn about the APIs and data structures. However, never make any changes in EAA's repository.

# EAA data structures

## Task manager

The workflow of a task should be wrapped in a subclass of `BaseTaskManager`. The workflow is not necessarily driven by an LLM agent: it might be a purely logic-driven or rule-based process, such as Bayesian optimization. This should be supported equally by EAA. 

## Tools

We use tools for both LLM-driven and logic-driven workflows to access and control experimental instruments, collect data, and move motors. Simulated tools and some data analysis tools are built-in inside this package as subclasses of `BaseTool`. Tools interacting with actual instruments are MCP servers which are not included in this repository. When using an MCP server, use the `MCPTool` in EAA to create a client.

When an MCP server is used by a logic-driven workflow where the tools are called as RPC calls in Python code instead of being called by an LLM, the `MCPTool` object should be wrapped by EAA's `MCPRPCWrapper` with proper tool name and tool argument mapping so that the task managers can stay agnostic to the actual names and arguments in the MCP server.

# Coding practices

- Use numpy/scipy docstring style, i.e.,
```
Parameters
----------
    a : int
    The input a.

Returns
-------
    int
    The output b.
```