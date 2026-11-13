import textwrap

from astprune.data.labels import align_edit, edit_matches_file, gold_for_file, parse_patch
from astprune.parse import parse_source

BEFORE = textwrap.dedent('''\
    import os

    LIMIT = 10


    def alpha(x):
        y = x + 1
        return y


    def beta(x):
        return x * 2


    class Box:
        size = 1

        def area(self):
            return self.size ** 2

        def volume(self):
            return self.size ** 3
    ''')


def patch_for(body: str) -> str:
    return "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n" + textwrap.dedent(body)


def gold(body: str):
    pf = parse_source(BEFORE, "pkg/mod.py")
    edit = parse_patch(patch_for(body))["pkg/mod.py"]
    assert edit_matches_file(pf, edit)
    g = gold_for_file(pf, edit)
    return pf, g, [pf.nodes[i].name for i in g.node_idx], [pf.nodes[i].name for i in g.function_idx]


def test_rewritten_line_marks_its_function():
    pf, g, nodes, funcs = gold('''\
        @@ -6,3 +6,3 @@
         def alpha(x):
        -    y = x + 1
        +    y = x + 2
             return y
        ''')
    assert funcs == ["alpha"] and g.lines == [7]


def test_insertion_inside_a_body_marks_the_function():
    pf, g, nodes, funcs = gold('''\
        @@ -6,3 +6,5 @@
         def alpha(x):
             y = x + 1
        +    if y < 0:
        +        y = 0
             return y
        ''')
    assert funcs == ["alpha"]
    assert g.lines == [7, 8]  # the two lines around the insertion point


def test_new_function_between_functions_marks_nothing():
    pf, g, nodes, funcs = gold('''\
        @@ -8,5 +8,9 @@
             return y


        +def gamma(x):
        +    return x
        +
        +
         def beta(x):
             return x * 2
        ''')
    assert nodes == [] and funcs == [] and g.lines == []


def test_new_function_right_after_last_line_marks_nothing():
    # Same edit as above, but the diff attaches the new code to alpha's last line.
    pf, g, nodes, funcs = gold('''\
        @@ -6,4 +6,8 @@
         def alpha(x):
             y = x + 1
             return y
        +
        +
        +def gamma(x):
        +    return x

        ''')
    assert funcs == []


def test_new_method_in_class_marks_nothing_but_changed_method_does():
    pf, g, nodes, funcs = gold('''\
        @@ -18,5 +18,8 @@
             def area(self):
        -        return self.size ** 2
        +        return self.size * self.size

        +    def perimeter(self):
        +        return 4 * self.size
        +
             def volume(self):
                 return self.size ** 3
        ''')
    assert funcs == ["Box.area"]


def test_class_attribute_edit_is_a_class_node_not_a_function():
    pf, g, nodes, funcs = gold('''\
        @@ -15,3 +15,3 @@
         class Box:
        -    size = 1
        +    size = 2

        ''')
    assert nodes == ["Box"] and funcs == []


def test_added_decorator_marks_the_function_below():
    pf, g, nodes, funcs = gold('''\
        @@ -9,5 +9,6 @@


        +@cache
         def beta(x):
             return x * 2

        ''')
    assert funcs == ["beta"]


def test_added_import_marks_the_import_block():
    pf, g, nodes, funcs = gold('''\
        @@ -1,3 +1,4 @@
         import os
        +import sys

         LIMIT = 10
        ''')
    assert funcs == [] and len(nodes) == 1 and pf.nodes[g.node_idx[0]].kind == "block"


def test_two_functions_in_one_patch():
    pf, g, nodes, funcs = gold('''\
        @@ -6,7 +6,7 @@
         def alpha(x):
        -    y = x + 1
        +    y = x - 1
             return y


         def beta(x):
        -    return x * 2
        +    return x * 3
        ''')
    assert funcs == ["alpha", "beta"]


def test_new_files_tests_and_non_python_are_skipped():
    patch = (
        "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/pkg/new.py b/pkg/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/pkg/new.py\n@@ -0,0 +1 @@\n+x = 1\n"
    )
    assert parse_patch(patch) == {}
    assert parse_patch("this is not a diff") == {}


def test_mismatched_patch_is_detected():
    pf = parse_source(BEFORE, "pkg/mod.py")
    edit = parse_patch(patch_for('''\
        @@ -6,3 +6,3 @@
         def alpha(x):
        -    y = x + 999
        +    y = x + 2
             return y
        '''))["pkg/mod.py"]
    assert not edit_matches_file(pf, edit)


def test_hunk_with_wrong_line_numbers_is_realigned():
    # Same change as the first test, but the header claims it starts 3 lines early.
    pf = parse_source(BEFORE, "pkg/mod.py")
    edit = parse_patch(patch_for('''\
        @@ -3,3 +3,3 @@
         def alpha(x):
        -    y = x + 1
        +    y = x + 2
             return y
        '''))["pkg/mod.py"]
    assert not edit_matches_file(pf, edit)
    aligned = align_edit(pf, edit)
    assert aligned is not None and edit_matches_file(pf, aligned)
    g = gold_for_file(pf, aligned)
    assert [pf.nodes[i].name for i in g.function_idx] == ["alpha"] and g.lines == [7]


def test_hunk_that_fits_nowhere_is_rejected():
    pf = parse_source(BEFORE, "pkg/mod.py")
    edit = parse_patch(patch_for('''\
        @@ -6,3 +6,3 @@
         def alpha(x):
        -    y = x + 999
        +    y = x + 2
             return y
        '''))["pkg/mod.py"]
    assert align_edit(pf, edit) is None


def test_multi_line_decorator_marks_the_function_below():
    pf, g, nodes, funcs = gold('''\
        @@ -9,5 +9,9 @@
         
         
        +@deprecated(
        +    since="2.0",
        +    alternative="gamma",
        +)
         def beta(x):
             return x * 2
         
        ''')
    assert funcs == ["beta"]


def test_new_decorated_function_marks_nothing():
    pf, g, nodes, funcs = gold('''\
        @@ -9,5 +9,10 @@
         
         
        +@cache
        +def gamma(x):
        +    return x
        +
        +
         def beta(x):
             return x * 2
         
        ''')
    assert funcs == []


def test_blank_line_replaced_by_code_marks_the_function():
    before = BEFORE.replace("    y = x + 1\n    return y\n", "    y = x + 1\n\n    return y\n")
    pf = parse_source(before, "pkg/mod.py")
    edit = parse_patch(patch_for('''\
        @@ -6,4 +6,4 @@
         def alpha(x):
             y = x + 1
        -
        +    y = abs(y)
             return y
        '''))["pkg/mod.py"]
    edit = align_edit(pf, edit)
    g = gold_for_file(pf, edit)
    assert [pf.nodes[i].name for i in g.function_idx] == ["alpha"]


def test_line_appended_to_a_body_just_before_a_rewritten_neighbour():
    # The new line ends first()'s body and the very next diff line rewrites second().
    # The added run touches removed code, but it still changes first().
    before = "def first():\n    return 1\ndef second():\n    return 2\n"
    pf = parse_source(before, "pkg/mod.py")
    edit = parse_patch(patch_for('''\
        @@ -1,4 +1,5 @@
         def first():
             return 1
        +    leftover = True
        -def second():
        +def second(x=0):
             return 2
        '''))["pkg/mod.py"]
    g = gold_for_file(pf, align_edit(pf, edit))
    assert [pf.nodes[i].name for i in g.function_idx] == ["first", "second"]


def test_line_appended_to_a_method_next_to_a_removed_blank_line():
    pf, g, nodes, funcs = gold('''\
        @@ -18,5 +18,6 @@
             def area(self):
                 return self.size ** 2
        +        self.cached = True
        -
        -    def volume(self):
        +
        +    def volume(self, depth=1):
                 return self.size ** 3
        ''')
    assert funcs == ["Box.area", "Box.volume"]


def test_attribute_added_to_a_class_block_that_starts_with_a_blank_line():
    before = BEFORE.replace("    def volume(self):", "    label = 'box'\n\n    def volume(self):")
    pf = parse_source(before, "pkg/mod.py")
    edit = parse_patch(patch_for('''\
        @@ -20,4 +20,5 @@
         
             label = 'box'
        +    colour = 'red'
         
             def volume(self):
        '''))["pkg/mod.py"]
    g = gold_for_file(pf, align_edit(pf, edit))
    assert len(g.node_idx) == 1 and pf.nodes[g.node_idx[0]].kind == "block" and g.function_idx == []


def test_continuation_line_added_inside_an_import_marks_the_block():
    before = "from pkg import (\n    alpha,\n    beta,\n)\n\nX = 1\n"
    pf = parse_source(before, "pkg/mod.py")
    edit = parse_patch(patch_for('''\
        @@ -1,4 +1,5 @@
         from pkg import (
             alpha,
        +    gamma,
             beta,
         )
        '''))["pkg/mod.py"]
    g = gold_for_file(pf, align_edit(pf, edit))
    assert len(g.node_idx) == 1 and g.lines
