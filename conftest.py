"""Puts the repo root on sys.path so `pytest` finds the top-level modules.

pytest adds the *test* directory to sys.path, not the project root, so without
this `import grid_delivery_robot` fails from a bare `pytest` invocation. A root
conftest.py is the standard fix for a flat, package-less layout like this one.
"""
