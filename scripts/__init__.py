"""Thin command-line entry points for the fnb project.

Scripts in this package contain **no research logic** — only argument parsing
and orchestration that delegates into ``fnb.*``. They mirror the direct
``python scripts/<name>.py`` invocations used inside the Kaggle runner notebook
and are also exposed as console entry points (see ``pyproject.toml``).
"""
