"""Locations shared by the standalone code repository and local delivery bundle."""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
_configured = os.environ.get('PCJBF_RESULTS_DIR')
_bundled = CODE_ROOT.parent / 'results'
RESULTS_ROOT = (Path(_configured).expanduser().resolve() if _configured else
                _bundled if (_bundled / 'run_manifest.json').is_file() else
                CODE_ROOT / 'results')
