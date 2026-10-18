"""Pickle-safe, observation-only forward counting for original ALL checkpoints."""


class PolicyForwardCounter:
    """No Runtime/provider references: torch.save(model) must remain supported.

    ALL periodically pickles its whole policy model, including registered hooks.
    A local closure (or a bound Runtime method) would capture non-pickleable run
    state. This importable callable stores only an optional cap and an integer.
    It never changes inputs, outputs, parameters, gradients or random state.
    """

    def __init__(self, limit=None):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError('Expected positive policy forward cap or None')
        self.limit = limit
        self.calls = 0

    def __call__(self, module, inputs):
        if self.limit is not None and self.calls >= self.limit:
            raise ValueError('Entry policy forward cap exceeded')
        self.calls += 1
