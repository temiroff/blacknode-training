# Policy Preview

Component of `blacknode-training`.

Node sources for this component belong in this folder. Until they move here,
nodes claim the component inline:

    @node(name="MyNode", component="policy-preview", ...)

Once sources live here, declare the folder in `blacknode-package.toml`:

    [components.policy-preview]
    nodes = ["components/policy-preview/nodes"]

and the inline `component=` argument can be dropped — the loader infers it
from the directory.
