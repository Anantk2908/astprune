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
