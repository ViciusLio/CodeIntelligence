# conftest.py -- pytest configuration for CodeIntelligence
#
# collect_context.py exports a function named `test_context` (a collector).
# Exclude it from pytest's test collection so pytest does not try to run it
# as a test fixture.

collect_ignore = ["collect_context.py"]
