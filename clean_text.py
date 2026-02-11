import re
import unicodedata

# -------- Common PDF broken word patterns --------
# Maps broken patterns to correct words
BROKEN_WORD_FIXES = {
    r'\bfi\s*le\b': 'file',
    r'\bfi\s*les\b': 'files',
    r'\bde\s*fi\s*ne\b': 'define',
    r'\bde\s*fi\s*ned\b': 'defined',
    r'\bde\s*fi\s*nes\b': 'defines',
    r'\bde\s*fi\s*ni\s*tion\b': 'definition',
    r'\beff\s*ective\b': 'effective',
    r'\beff\s*ect\b': 'effect',
    r'\barch\s*i\s*tec\s*ture\b': 'architecture',
    r'\barch\s*i\s*tec\s*tures\b': 'architectures',
    r'\bin\s*struc\s*tion\b': 'instruction',
    r'\bin\s*struc\s*tions\b': 'instructions',
    r'\bpro\s*gram\b': 'program',
    r'\bpro\s*grams\b': 'programs',
    r'\bpro\s*gram\s*ming\b': 'programming',
    r'\bmem\s*ory\b': 'memory',
    r'\breg\s*is\s*ter\b': 'register',
    r'\breg\s*is\s*ters\b': 'registers',
    r'\bcom\s*puter\b': 'computer',
    r'\bcom\s*puters\b': 'computers',
    r'\bex\s*ample\b': 'example',
    r'\bex\s*amples\b': 'examples',
    r'\bex\s*e\s*cute\b': 'execute',
    r'\bex\s*e\s*cution\b': 'execution',
    r'\boper\s*a\s*tion\b': 'operation',
    r'\boper\s*a\s*tions\b': 'operations',
    r'\boper\s*and\b': 'operand',
    r'\boper\s*ands\b': 'operands',
    r'\bper\s*for\s*mance\b': 'performance',
    r'\bproc\s*ess\s*or\b': 'processor',
    r'\bproc\s*ess\s*ors\b': 'processors',
    r'\bad\s*dress\b': 'address',
    r'\bad\s*dresses\b': 'addresses',
    r'\bad\s*dress\s*ing\b': 'addressing',
    r'\bla\s*bel\b': 'label',
    r'\bla\s*bels\b': 'labels',
    r'\bseg\s*ment\b': 'segment',
    r'\bseg\s*ments\b': 'segments',
    r'\bsec\s*tion\b': 'section',
    r'\bsec\s*tions\b': 'sections',
    r'\bfunc\s*tion\b': 'function',
    r'\bfunc\s*tions\b': 'functions',
    r'\bval\s*ue\b': 'value',
    r'\bval\s*ues\b': 'values',
    r'\bnum\s*ber\b': 'number',
    r'\bnum\s*bers\b': 'numbers',
    r'\bsys\s*tem\b': 'system',
    r'\bsys\s*tems\b': 'systems',
    r'\bdi\s*rect\b': 'direct',
    r'\bim\s*me\s*di\s*ate\b': 'immediate',
    r'\bim\s*ple\s*ment\b': 'implement',
    r'\bim\s*ple\s*men\s*ta\s*tion\b': 'implementation',
    r'\bcon\s*trol\b': 'control',
    r'\bcon\s*tains\b': 'contains',
    r'\bcon\s*di\s*tion\b': 'condition',
    r'\bcon\s*di\s*tions\b': 'conditions',
    r'\bcon\s*di\s*tion\s*al\b': 'conditional',
    r'\bbranch\s*ing\b': 'branching',
    r'\bloop\s*ing\b': 'looping',
    r'\bex\s*cep\s*tion\b': 'exception',
    r'\bex\s*cep\s*tions\b': 'exceptions',
    r'\bin\s*ter\s*rupt\b': 'interrupt',
    r'\bin\s*ter\s*rupts\b': 'interrupts',
    r'\bfl\s*oat\b': 'float',
    r'\bfl\s*oat\s*ing\b': 'floating',
    r'\bpi\s*pe\s*line\b': 'pipeline',
    r'\bpi\s*pe\s*lin\s*ing\b': 'pipelining',
    r'\bca\s*che\b': 'cache',
    r'\bca\s*ches\b': 'caches',
    r'\bca\s*ch\s*ing\b': 'caching',
    r'\bpar\s*al\s*lel\b': 'parallel',
    r'\bpar\s*al\s*lel\s*ism\b': 'parallelism',
    r'\bhard\s*ware\b': 'hardware',
    r'\bsoft\s*ware\b': 'software',
    r'\bas\s*sem\s*bly\b': 'assembly',
    r'\bas\s*sem\s*bler\b': 'assembler',
    r'\bas\s*sem\s*blers\b': 'assemblers',
    r'\bcom\s*piler\b': 'compiler',
    r'\bcom\s*pilers\b': 'compilers',
    r'\blink\s*er\b': 'linker',
    r'\blink\s*ers\b': 'linkers',
    r'\bload\s*er\b': 'loader',
    r'\bload\s*ers\b': 'loaders',
    r'\bbin\s*ary\b': 'binary',
    r'\bhex\s*a\s*dec\s*i\s*mal\b': 'hexadecimal',
    r'\bdec\s*i\s*mal\b': 'decimal',
    r'\boc\s*tal\b': 'octal',
    r'\bbyte\s*s\b': 'bytes',
    r'\bbit\s*s\b': 'bits',
    r'\bop\s*code\b': 'opcode',
    r'\bop\s*codes\b': 'opcodes',
    # Common OCR errors
    r'\ba!\s*er\b': 'after',
    r'\baft\s*er\b': 'after',
    r'\bbef\s*ore\b': 'before',
    r'\bwh\s*ich\b': 'which',
    r'\bthr\s*ough\b': 'through',
}

# -------- Core Cleaning --------

def normalize_unicode(text):
    # normalize unicode and remove non-printable chars
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text

def fix_broken_words(text):
    """Fix known PDF extraction broken word patterns."""
    for pattern, replacement in BROKEN_WORD_FIXES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def fix_hyphenated_linebreaks(text):
    """Fix words split across lines with hyphens: 'lan-\\nguage' -> 'language'"""
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    return text

def remove_garbage_symbols(text):
    # remove leftover junk characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text

def remove_figures_and_captions(text):
    lines = []
    for line in text.split("\n"):
        if re.match(r"^FIGURE\s+[A-Z0-9.]+", line):
            continue
        if re.match(r"^Appendix\s+[A-Z]", line):
            continue
        if re.match(r"^[A-Z]-\d+", line):  # page numbers like B-55
            continue
        lines.append(line)
    return "\n".join(lines)

def remove_references_section(text):
    # cut off at References / Bibliography
    split = re.split(r"\n(References|Bibliography)\n", text, flags=re.IGNORECASE)
    return split[0]

def collapse_whitespace(text):
    # normalize whitespace
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

# -------- Main API --------

def clean_chapter_text(raw_text):
    text = raw_text

    text = normalize_unicode(text)
    text = fix_hyphenated_linebreaks(text)  # Fix lan-\nguage -> language
    text = fix_broken_words(text)           # Fix fi le -> file
    text = remove_garbage_symbols(text)
    text = remove_figures_and_captions(text)
    text = remove_references_section(text)
    text = collapse_whitespace(text)

    return text
