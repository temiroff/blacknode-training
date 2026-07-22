# Training Jobs

Component of `blacknode-training`.

Node sources for this component belong in this folder. Until they move here,
nodes claim the component inline:

    @node(name="MyNode", component="training-jobs", ...)

Once sources live here, declare the folder in `blacknode-package.toml`:

    [components.training-jobs]
    nodes = ["components/training-jobs/nodes"]

and the inline `component=` argument can be dropped — the loader infers it
from the directory.
