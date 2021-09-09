import sys
import traceback
from pathlib import Path

# cpu vs gpu
try:
    from classes import classes
    from integration import Integration

    integration = Integration()
    integration.load(None, {"classes": classes})
    print(integration.infer([Path("test.jpg")], {}))
    print("Success!")
except Exception as e:
    traceback.print_exception(type(e), e, e.__traceback__)
    sys.exit(1)
