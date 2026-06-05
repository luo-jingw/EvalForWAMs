# Package init for the lingbot-va WAM adapter.
#
# Sibling repo `lingbot-va/` is not pip-installed; we expose its source tree
# on sys.path so that `from wan_va.modules.model import WanTransformerBlock`
# and similar imports work from anywhere inside this package.
import os
import sys

LINGBOT_VA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "lingbot-va")
)

if LINGBOT_VA_PATH not in sys.path:
    sys.path.insert(0, LINGBOT_VA_PATH)
