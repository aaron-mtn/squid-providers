"""Providers import only small Core contracts, never the application."""

import subprocess
import sys


def test_provider_import_dependencies_and_cancellation_identity():
    script = """
import sys, builtins
class BlockApplication:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "code_squid":
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockApplication())
original_import = builtins.__import__
def checked_import(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {}).get("__name__", "")
    if caller.startswith("squid_providers") and name.startswith("squid_core"):
        assert name in {"squid_core.model_response", "squid_core.providers.base"}, (caller, name)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = checked_import
import squid_providers.catalog
from squid_core.providers.base import BaseProvider, StreamCancelled
from squid_providers.cancellation import StreamCancelled as Cancelled
from squid_providers.ollama import OllamaProvider
from squid_providers.openai_compatible import OpenAICompatibleProvider
assert Cancelled is StreamCancelled
assert issubclass(OllamaProvider, BaseProvider)
assert issubclass(OpenAICompatibleProvider, BaseProvider)
"""
    subprocess.run([sys.executable, "-c", script], check=True)
