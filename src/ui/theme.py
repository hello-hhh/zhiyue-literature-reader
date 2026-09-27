"""ui/theme.py — CSS + style constants for the Streamlit UI.

WHY THIS FILE EXISTS:
    All visual styling — fonts, colors, spacing, card styles, badges — is
    defined here as a single CSS string. Keeping it separate from handlers.py
    means the visual design can be changed in one place without touching any
    logic. app.py injects this CSS once on startup via st.markdown().

DESIGN SYSTEM:
    Font:       IBM Plex Sans (body) + IBM Plex Mono (code, labels, badges)
    Palette:    Blue (#1a56db primary) on near-white (#f9fafb) background
    Radius:     6 px for inputs/badges, 10 px for cards and chat bubbles
    Shadow:     0 2px 8px rgba(0,0,0,.06) for cards — gives depth without noise
    Focus:      2 px blue outline ring — consistent across all interactive elements
"""

__all__ = ['CSS', 'BADGE_CLASSES', 'CONFIDENCE_BADGE', 'AVATAR']

# ── CSS ────────────────────────────────────────────────────────────────────────
# Complete stylesheet injected into the Streamlit page head via st.markdown().
# Organised top-to-bottom: fonts → tokens → base → header → chat →
# sidebar → badges → cards → buttons → inputs → expanders → misc.

CSS: str = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Design tokens — Sea Blue + Teal Green palette ──────────────────────────── */
:root {
    /* Primary: deep ocean blue → sea teal gradient */
    --blue-700:   #0e7490;   /* deep cyan-700 */
    --blue-600:   #0891b2;   /* sea blue cyan-600 */
    --blue-500:   #22d3ee;   /* bright cyan-400 */
    --blue-100:   #cffafe;   /* pale cyan */
    --blue-50:    #ecfeff;   /* whisper cyan */

    /* Accent: seafoam / teal green */
    --teal-700:   #0f766e;   /* deep teal */
    --teal-600:   #0d9488;   /* seafoam teal-600 */
    --teal-100:   #ccfbf1;   /* pale teal */

    --gray-900:   #0f172a;   /* deep slate — richer than pure black */
    --gray-800:   #1e293b;
    --gray-700:   #334155;
    --gray-500:   #64748b;
    --gray-400:   #94a3b8;
    --gray-200:   #e2e8f0;
    --gray-100:   #f1f5f9;
    --gray-50:    #f8fafc;   /* barely-there aqua background */

    --green-700:  #0f766e;   /* reuse teal for "ok/pass" badges */
    --green-100:  #ccfbf1;
    --green-50:   #f0fdfa;

    --amber-700:  #b45309;
    --amber-100:  #fef3c7;

    --purple-700: #6d28d9;
    --purple-100: #ede9fe;

    --white:      #ffffff;
    --shadow-sm:  0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
    --shadow-md:  0 4px 12px rgba(0,0,0,.08), 0 2px 4px rgba(0,0,0,.05);
    --radius-sm:  6px;
    --radius-md:  10px;
    --radius-lg:  14px;
}

/* ── Base ──────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    background:  var(--gray-50);
    color:       var(--gray-900);
    font-size:   15px;
    line-height: 1.6;
}
.stApp { background: var(--gray-50) !important; }

/* ── App header ────────────────────────────────────────────────────────────── */
/* Product-quality header with a gradient accent bar above the title */
.rag-header {
    background:    var(--white);
    border:        1px solid var(--gray-200);
    border-radius: var(--radius-md);
    padding:       1rem 1.25rem 0.85rem;
    margin-bottom: 1.25rem;
    box-shadow:    var(--shadow-sm);
    position:      relative;
    overflow:      hidden;
}
/* Sea-blue → teal-green gradient accent bar — the product brand stripe */
.rag-header::before {
    content:    '';
    position:   absolute;
    top:        0; left: 0; right: 0;
    height:     3px;
    background: linear-gradient(90deg, var(--blue-600), var(--teal-600));
    border-radius: var(--radius-md) var(--radius-md) 0 0;
}
.rag-title {
    font-family:    'IBM Plex Mono', monospace;
    font-size:      1.6rem;
    font-weight:    600;
    color:          var(--blue-600);
    letter-spacing: -.025em;
    margin:         0 0 .2rem;
    line-height:    1.2;
}
.rag-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size:   .68rem;
    color:       var(--gray-400);
    letter-spacing: .01em;
    margin:      0;
}
/* Capability chips after the subtitle */
.rag-chips {
    display:    flex;
    flex-wrap:  wrap;
    gap:        .3rem;
    margin-top: .55rem;
}
.rag-chip {
    display:       inline-block;
    font-family:   'IBM Plex Mono', monospace;
    font-size:     .62rem;
    font-weight:   600;
    padding:       .15rem .45rem;
    border-radius: 4px;
    background:    var(--blue-50);
    color:         var(--blue-600);
    border:        1px solid var(--blue-100);
    letter-spacing: .03em;
}

/* ── Mode selector ──────────────────────────────────────────────────────────── */
/* Wrap the radio in a subtle card so it looks like a tab bar */
.stRadio > div {
    background:    var(--white) !important;
    border:        1px solid var(--gray-200) !important;
    border-radius: var(--radius-sm) !important;
    padding:       .35rem .6rem !important;
    box-shadow:    var(--shadow-sm) !important;
    display:       inline-flex !important;
    gap:           .4rem !important;
}
.stRadio label { font-weight: 500 !important; color: var(--gray-700) !important; }

/* ── Chat bubbles ───────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background:    var(--white);
    border:        1px solid var(--gray-200);
    border-radius: var(--radius-md);
    margin:        .4rem 0;
    box-shadow:    var(--shadow-sm);
    transition:    box-shadow .15s ease;
}
[data-testid="stChatMessage"]:hover { box-shadow: var(--shadow-md); }
[data-testid="stChatMessageContent"] p { margin: 0; line-height: 1.65; }

/* User message — subtle blue tint to distinguish from assistant */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: var(--blue-50);
    border-color: var(--blue-100);
}

/* ── Agent step cards ────────────────────────────────────────────────────────── */
.step {
    background:    var(--green-50);
    border:        1px solid var(--green-100);
    border-left:   3px solid var(--green-700);
    border-radius: var(--radius-sm);
    padding:       .45rem .7rem;
    margin:        .2rem 0;
    font-family:   'IBM Plex Mono', monospace;
    font-size:     .72rem;
    color:         var(--green-700);
    line-height:   1.5;
}

/* ── Agent tools reference panel ─────────────────────────────────────────────── */
.tools-panel {
    background:    var(--blue-50);
    border:        1px solid var(--blue-100);
    border-left:   3px solid var(--blue-600);
    border-radius: var(--radius-sm);
    padding:       .75rem 1rem;
    margin:        .5rem 0;
    font-size:     .84rem;
    line-height:   1.75;
}
.tools-panel b { color: var(--blue-700); }

/* ── Chunk / retrieval cards ─────────────────────────────────────────────────── */
.chunk {
    background:    var(--white);
    border:        1px solid var(--gray-200);
    border-radius: var(--radius-sm);
    padding:       .45rem .65rem;
    margin:        .2rem 0;
    font-family:   'IBM Plex Mono', monospace;
    font-size:     .7rem;
    color:         var(--gray-700);
    line-height:   1.5;
    box-shadow:    var(--shadow-sm);
}
/* Score label — bold blue number */
.cs  { color: var(--blue-600); font-weight: 600; }
/* Source path — muted blue */
.src { color: var(--blue-500); }

/* ── Badges ──────────────────────────────────────────────────────────────────── */
.badge {
    display:        inline-flex;
    align-items:    center;
    font-family:    'IBM Plex Mono', monospace;
    font-size:      .63rem;
    font-weight:    600;
    padding:        .18rem .5rem;
    border-radius:  20px;   /* pill shape */
    margin:         .1rem .15rem .1rem 0;
    letter-spacing: .03em;
    border:         1px solid transparent;
}
/* Query type badges */
.b-fact { background: var(--blue-50);    color: var(--blue-700);   border-color: var(--blue-100); }
.b-comp { background: var(--purple-100); color: var(--purple-700); border-color: #c4b5fd; }
.b-gen  { background: var(--gray-100);   color: var(--gray-700);   border-color: var(--gray-200); }
/* Confidence badges */
.b-ok   { background: var(--green-100);  color: var(--green-700);  border-color: #bbf7d0; }
.b-warn { background: var(--amber-100);  color: var(--amber-700);  border-color: #fde68a; }

/* ── Sidebar ────────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background:   var(--white) !important;
    border-right: 1px solid var(--gray-200) !important;
}
/* Sidebar section label */
.sidebar-section {
    font-family:    'IBM Plex Mono', monospace;
    font-size:      .68rem;
    font-weight:    600;
    color:          var(--gray-400);
    letter-spacing: .08em;
    text-transform: uppercase;
    margin:         .8rem 0 .3rem;
}

/* ── Sidebar stats ───────────────────────────────────────────────────────────── */
.stat {
    font-family:     'IBM Plex Sans', sans-serif;
    font-size:       .82rem;
    color:           var(--gray-600, #4b5563);
    padding:         .32rem 0;
    border-bottom:   1px solid var(--gray-100);
    display:         flex;
    justify-content: space-between;
    align-items:     center;
}
/* Stat value — prominent blue number */
.sv {
    font-family: 'IBM Plex Mono', monospace;
    font-size:   .82rem;
    color:       var(--blue-600);
    font-weight: 600;
}

/* ── Buttons ────────────────────────────────────────────────────────────────── */
/* Primary — solid blue */
.stButton > button {
    background:    var(--blue-600) !important;
    color:         var(--white) !important;
    font-family:   'IBM Plex Sans', sans-serif !important;
    font-weight:   500 !important;
    font-size:     .88rem !important;
    border:        none !important;
    border-radius: var(--radius-sm) !important;
    padding:       .45rem 1rem !important;
    transition:    background .15s ease, transform .1s ease, box-shadow .15s ease !important;
    box-shadow:    0 1px 2px rgba(8,145,178,.3) !important;
    letter-spacing: .01em !important;
}
.stButton > button:hover {
    background: var(--blue-700) !important;
    box-shadow: 0 3px 8px rgba(8,145,178,.35) !important;
    transform:  translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Form submit button (slightly different style to distinguish from actions) */
[data-testid="stFormSubmitButton"] > button {
    background:    var(--blue-600) !important;
    color:         var(--white) !important;
    font-family:   'IBM Plex Mono', monospace !important;
    font-weight:   600 !important;
    border-radius: var(--radius-sm) !important;
    letter-spacing: .02em !important;
}

/* ── Expanders (URL, file upload panels) ────────────────────────────────────── */
[data-testid="stExpander"] {
    background:    var(--white) !important;
    border:        1px solid var(--gray-200) !important;
    border-radius: var(--radius-md) !important;
    margin-bottom: .6rem !important;
    box-shadow:    var(--shadow-sm) !important;
    overflow:      hidden !important;
}
[data-testid="stExpander"] > div:first-child {
    background: var(--gray-50) !important;
    border-bottom: 1px solid var(--gray-200) !important;
    padding: .55rem .85rem !important;
}

/* ── Text inputs ────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
textarea {
    font-family:   'IBM Plex Sans', sans-serif !important;
    background:    var(--white) !important;
    border:        1px solid var(--gray-200) !important;
    border-radius: var(--radius-sm) !important;
    color:         var(--gray-900) !important;
    font-size:     .9rem !important;
    transition:    border-color .15s ease, box-shadow .15s ease !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
textarea:focus {
    border-color: var(--blue-600) !important;
    box-shadow:   0 0 0 3px rgba(8,145,178,.15) !important;
    outline:      none !important;
}

/* ── Chat input ─────────────────────────────────────────────────────────────── */
.stChatInputContainer {
    background:    var(--white) !important;
    border:        1px solid var(--gray-200) !important;
    border-radius: var(--radius-md) !important;
    box-shadow:    var(--shadow-sm) !important;
}
.stChatInputContainer textarea {
    font-family:   'IBM Plex Sans', sans-serif !important;
    font-size:     .95rem !important;
    border:        none !important;
    background:    transparent !important;
}
[data-testid="stChatInputContainer"]:focus-within {
    border-color: var(--blue-600) !important;
    box-shadow:   0 0 0 3px rgba(8,145,178,.15) !important;
}

/* ── Checkboxes ─────────────────────────────────────────────────────────────── */
.stCheckbox label { font-size: .88rem !important; color: var(--gray-700) !important; }

/* ── Info / caption text ────────────────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    color:     var(--gray-500) !important;
    font-size: .82rem !important;
}

/* ── Dividers ───────────────────────────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid var(--gray-200) !important; margin: .75rem 0 !important; }

/* ── Progress bar ───────────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: var(--blue-600) !important;
    border-radius: 4px !important;
}
/* Progress label text — matches page backgroundColor from config.toml (#ffffff) */
[data-testid="stProgress"] p,
[data-testid="stProgress"] span,
[data-testid="stProgress"] small {
    background: #ffffff !important;
}

/* ── Info boxes / notes ─────────────────────────────────────────────────────── */
/* Used for tips and guidance notes rendered via st.markdown */
.note {
    background:    var(--blue-50);
    border:        1px solid var(--blue-100);
    border-left:   3px solid var(--blue-500);
    border-radius: var(--radius-sm);
    padding:       .55rem .8rem;
    font-size:     .84rem;
    color:         var(--blue-700);
    margin:        .4rem 0 .6rem;
    line-height:   1.55;
}
/* Warning note variant */
.note-warn {
    background:  var(--amber-100);
    border-color: #fde68a;
    border-left-color: var(--amber-700);
    color:        var(--amber-700);
}

/* ── Global focus ring ──────────────────────────────────────────────────────── */
*:focus-visible { outline: 2px solid var(--blue-600) !important; outline-offset: 2px !important; }
:root { --primary-color: var(--blue-600) !important; }

/* ── Scrollbar (WebKit) ─────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--gray-100); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: var(--gray-300, #d1d5db); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gray-400); }
</style>
"""

# Minimal reader-specific layer. It intentionally uses system fonts, quiet
# neutrals, one ink-blue accent, and almost no decoration.
CSS += """
<style>
:root {
    --reader-ink: #1f2933;
    --reader-muted: #6b7280;
    --reader-line: #e7e7e2;
    --reader-paper: #ffffff;
    --reader-canvas: #f7f7f4;
    --reader-accent: #315b78;
}
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    color: var(--reader-ink);
}
.stApp { background: var(--reader-canvas) !important; }
[data-testid="stSidebar"] { background: #f1f1ed !important; }
.block-container { padding-top: 2rem; max-width: 1500px; }
.brand { font-size: 1.55rem; font-weight: 700; letter-spacing: .08em; margin-top: .4rem; }
.source-note { color: var(--reader-muted); font-size: .75rem; margin-top: 2rem; }
.reader-header {
    display: flex; align-items: end; justify-content: space-between;
    padding: 0 0 1.1rem; margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--reader-line);
}
.reader-title { font-size: 1.8rem; font-weight: 700; letter-spacing: .08em; }
.reader-subtitle { color: var(--reader-muted); font-size: .9rem; margin-top: .2rem; }
.reader-formats { color: var(--reader-muted); font-size: .78rem; letter-spacing: .08em; }
.empty-state {
    text-align: center; color: var(--reader-muted); padding: 5rem 2rem;
    border: 1px dashed #d6d6d0; border-radius: 8px; background: rgba(255,255,255,.55);
}
.empty-title { color: var(--reader-ink); font-size: 1.05rem; font-weight: 600; margin-bottom: .4rem; }
.assistant-empty {
    color: var(--reader-muted); padding: 2rem 1rem; text-align: center;
    border: 1px dashed var(--reader-line); border-radius: 8px;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--reader-paper); border-color: var(--reader-line) !important;
    box-shadow: none !important;
}
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background: var(--reader-accent) !important; border-radius: 6px !important;
    box-shadow: none !important; transform: none !important;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    background: #24475f !important; transform: none !important; box-shadow: none !important;
}
[data-testid="stChatMessage"] { box-shadow: none; border-color: var(--reader-line); }
[data-testid="stExpander"] { box-shadow: none !important; border-color: var(--reader-line) !important; }
[data-testid="stMetric"] { background: transparent; }

/* Remove Streamlit's English deployment/menu chrome from the reader UI. */
#MainMenu,
[data-testid="stToolbar"],
[data-testid="stAppDeployButton"],
[data-testid="stDecoration"],
footer { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* Localise the built-in uploader's remaining visible labels. */
[data-testid="stFileUploaderDropzoneInstructions"] > div { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"]::before {
    content: "拖放 PDF 或 Word 文件到这里";
    display: block;
    color: var(--reader-ink);
    font-size: .9rem;
}
[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: "单个文件最大 200MB";
    display: block;
    margin-top: .2rem;
    color: var(--reader-muted);
    font-size: .75rem;
}
[data-testid="stFileUploaderDropzone"] button { font-size: 0 !important; }
[data-testid="stFileUploaderDropzone"] button::after {
    content: "选择文件";
    font-size: .88rem;
}
.pdf-page-label { text-align:center; color:var(--reader-muted); font-size:.86rem; padding:.45rem 0; }
</style>
"""

# ── Badge CSS class mapping ────────────────────────────────────────────────────
# Maps the query_type string returned by the pipeline to a CSS badge class.

BADGE_CLASSES: dict = {
    'factual':    'b-fact',
    'comparison': 'b-comp',
    'general':    'b-gen',
    'summarise':  'b-gen',
}

# ── Confidence badge CSS class ─────────────────────────────────────────────────
# True  = confident answer found,  False = low-confidence or no-info response.

CONFIDENCE_BADGE: dict = {
    True:  'b-ok',
    False: 'b-warn',
}

# ── Chat avatar glyphs ────────────────────────────────────────────────────────
# Shown next to each chat message to identify the speaker at a glance.

AVATAR: dict = {
    'user':      '🧑',
    'assistant': '💬',
    'agent':     '🤖',
}
