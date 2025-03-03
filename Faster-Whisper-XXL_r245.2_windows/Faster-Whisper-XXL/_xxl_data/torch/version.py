from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']
__version__ = '2.2.2+cu121'
debug = False
cuda: Optional[str] = '12.1'
git_version = '39901f229520a5256505ec24782f716ee7ddc843'
hip: Optional[str] = None
