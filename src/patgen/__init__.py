"""patgen: mine SMS templates from messy bilingual text, then match them fast."""

from .drain import DrainTree
from .learn import LearnConfig, TemplateLearner, learn_templates, prepare
from .lexicon import available as list_lexicons
from .lexicon import load as load_lexicon
from .matcher import MatchResult, TemplateMatcher
from .normalize import NormalizeConfig, normalize, repair_mojibake
from .template import Literal, Slot, Template, TemplateLibrary
from .tokenizer import tokenize

__all__ = [
    "DrainTree",
    "LearnConfig",
    "Literal",
    "MatchResult",
    "NormalizeConfig",
    "Slot",
    "Template",
    "TemplateLearner",
    "TemplateLibrary",
    "TemplateMatcher",
    "learn_templates",
    "list_lexicons",
    "load_lexicon",
    "normalize",
    "prepare",
    "repair_mojibake",
    "tokenize",
]

__version__ = "0.1.0"
