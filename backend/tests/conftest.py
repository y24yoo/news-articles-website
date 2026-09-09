import sys
import types

# fred.py does `from api_key import fred_api` etc. api_key.py is a local,
# gitignored secrets file (see README) that never exists in CI or a fresh
# clone. Stub it so importing fred.py doesn't require real credentials.
_fake_api_key = types.ModuleType("api_key")
_fake_api_key.fred_api = "test-fred-key"
_fake_api_key.daytona_api = "test-daytona-key"
_fake_api_key.firecrawl_api = "test-firecrawl-key"
sys.modules["api_key"] = _fake_api_key
