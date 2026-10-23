"""Root conftest: ensure the project root is on sys.path for all test modules."""
import sys
from pathlib import Path

# Insert project root so that `import dataset_prep`, `import aggregate`, etc.
# work regardless of which directory pytest is invoked from.
_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
