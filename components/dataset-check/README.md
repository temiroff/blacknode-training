# Dataset Check

Component of `blacknode-training`.

Node sources for this component belong in this folder. Until they move here,
nodes claim the component inline:

    @node(name="MyNode", component="dataset-check", ...)

Once sources live here, declare the folder in `blacknode-package.toml`:

    [components.dataset-check]
    nodes = ["components/dataset-check/nodes"]

and the inline `component=` argument can be dropped — the loader infers it
from the directory.
