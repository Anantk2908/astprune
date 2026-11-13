import textwrap

import pytest

SAMPLE = textwrap.dedent('''\
    """Module docstring."""
    import os
    import sys

    CONSTANT = 3


    def top_level(a, b=2):
        """Add two numbers."""
        total = a + b
        return total


    @decorator
    def decorated(
        x,
        y,
    ) -> int:
        def inner():
            return x
        return inner() + y


    class Shape:
        """A shape."""

        sides = 0

        def area(self):
            return 0

        label = "shape"

        @property
        def name(self):
            return self.label

        class Meta:
            ordering = ["name"]

            def describe(self):
                return "meta"


    class Empty: pass


    def one_liner(): return 1


    async def fetch(url):
        return await get(url)
    ''')


@pytest.fixture
def sample_source() -> str:
    return SAMPLE
