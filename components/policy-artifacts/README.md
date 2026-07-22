# Policy Artifacts

Component of `blacknode-training`.

Node sources for this component belong in this folder. Until they move here,
nodes claim the component inline:

    @node(name="MyNode", component="policy-artifacts", ...)

Once sources live here, declare the folder in `blacknode-package.toml`:

    [components.policy-artifacts]
    nodes = ["components/policy-artifacts/nodes"]

and the inline `component=` argument can be dropped — the loader infers it
from the directory.
