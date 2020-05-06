#__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
from .spell import Spell
from .keyboardspell import KeyboardSpell
from .phoneticspell import PhoneticSpell
