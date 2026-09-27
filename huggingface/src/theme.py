"""theme.py — CSS design-system constant for the HF Space Gradio UI.

WHY THIS FILE EXISTS:
    The CSS string is ~215 lines on its own. Keeping it in handlers.py or
    ui_builder.py would push both past the 500-line limit and bury the design
    tokens inside application logic. A dedicated module makes it easy to
    update styling without touching any handler or wiring code.

USAGE:
    from src.theme import CSS
    with gr.Blocks(css=CSS, ...) as demo:
        ...
"""

__all__ = ['CSS']

# Full design-system CSS — matches the Streamlit version token-for-token.
# Same IBM Plex font stack, same color tokens, same shadow/radius values.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Design tokens — Sea Blue + Teal Green palette ──────────────────────────── */
:root {
    --blue-700:   #0e7490;   /* deep cyan-700 */
    --blue-600:   #0891b2;   /* sea blue */
    --blue-500:   #22d3ee;   /* bright cyan */
    --blue-100:   #cffafe;   /* pale cyan */
    --blue-50:    #ecfeff;   /* whisper cyan */
    --teal-700:   #0f766e;
    --teal-600:   #0d9488;   /* seafoam teal */
    --teal-100:   #ccfbf1;
    --gray-900:   #0f172a;
    --gray-700:   #334155;
    --gray-500:   #64748b;
    --gray-400:   #94a3b8;
    --gray-200:   #e2e8f0;
    --gray-100:   #f1f5f9;
    --gray-50:    #f8fafc;
    --green-700:  #0f766e;
    --green-100:  #ccfbf1;
    --amber-700:  #b45309;
    --amber-100:  #fef3c7;
    --white:      #ffffff;
    --shadow-sm:  0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
    --shadow-md:  0 4px 12px rgba(0,0,0,.08), 0 2px 4px rgba(0,0,0,.05);
    --radius-sm:  6px;
    --radius-md:  10px;
}

/* ── Base ──────────────────────────────────────────────────────────────────── */
.gradio-container {
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
    background: var(--gray-50) !important;
    color: var(--gray-900) !important;
    font-size: 15px;
    line-height: 1.6;
}

/* ── App header ────────────────────────────────────────────────────────────── */
/* Product-quality header card with a blue gradient accent bar at the top */
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
.rag-header::before {
    content:    '';
    position:   absolute;
    top: 0; left: 0; right: 0;
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
    font-family:    'IBM Plex Mono', monospace;
    font-size:      .68rem;
    color:          var(--gray-400);
    letter-spacing: .01em;
    margin:         0 0 .5rem;
}
/* Capability chips */
.rag-chips { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .5rem; }
.rag-chip {
    display:        inline-block;
    font-family:    'IBM Plex Mono', monospace;
    font-size:      .62rem;
    font-weight:    600;
    padding:        .15rem .45rem;
    border-radius:  4px;
    background:     var(--blue-50);
    color:          var(--blue-600);
    border:         1px solid var(--blue-100);
    letter-spacing: .03em;
}

/* ── Chatbot messages ──────────────────────────────────────────────────────── */
.message-wrap .message {
    font-family:   'IBM Plex Sans', sans-serif !important;
    font-size:     .92rem !important;
    line-height:   1.65 !important;
    border-radius: var(--radius-md) !important;
    box-shadow:    var(--shadow-sm) !important;
}
.message-wrap .user  { background: var(--blue-50)  !important; border: 1px solid var(--blue-100)  !important; }
.message-wrap .bot   { background: var(--white)     !important; border: 1px solid var(--gray-200)  !important; }

/* ── Buttons ───────────────────────────────────────────────────────────────── */
button.primary {
    background:     var(--blue-600) !important;
    color:          var(--white) !important;
    font-family:    'IBM Plex Sans', sans-serif !important;
    font-weight:    500 !important;
    font-size:      .88rem !important;
    border:         none !important;
    border-radius:  var(--radius-sm) !important;
    box-shadow:     0 1px 2px rgba(8,145,178,.3) !important;
    transition:     background .15s ease, transform .1s ease, box-shadow .15s ease !important;
}
button.primary:hover {
    background: var(--blue-700) !important;
    box-shadow: 0 3px 8px rgba(8,145,178,.35) !important;
    transform:  translateY(-1px) !important;
}
button.primary:active { transform: translateY(0) !important; }
button.secondary {
    background:    var(--white) !important;
    color:         var(--blue-600) !important;
    border:        1px solid var(--blue-600) !important;
    font-family:   'IBM Plex Sans', sans-serif !important;
    font-weight:   500 !important;
    border-radius: var(--radius-sm) !important;
    transition:    background .15s ease, transform .1s ease !important;
}
button.secondary:hover {
    background: var(--blue-50) !important;
    transform:  translateY(-1px) !important;
}

/* ── Accordions / panel cards ──────────────────────────────────────────────── */
.accordion {
    background:    var(--white) !important;
    border:        1px solid var(--gray-200) !important;
    border-radius: var(--radius-md) !important;
    box-shadow:    var(--shadow-sm) !important;
    margin-bottom: .6rem !important;
    overflow:      hidden !important;
}

/* ── Text inputs ───────────────────────────────────────────────────────────── */
input[type="text"], input[type="number"], textarea {
    font-family:   'IBM Plex Sans', sans-serif !important;
    background:    var(--white) !important;
    border:        1px solid var(--gray-200) !important;
    border-radius: var(--radius-sm) !important;
    color:         var(--gray-900) !important;
    font-size:     .9rem !important;
    transition:    border-color .15s ease, box-shadow .15s ease !important;
}
input[type="text"]:focus, input[type="number"]:focus, textarea:focus {
    border-color: var(--blue-600) !important;
    box-shadow:   0 0 0 3px rgba(8,145,178,.15) !important;
    outline:      none !important;
}

/* ── Info note boxes ───────────────────────────────────────────────────────── */
/* Used for tips rendered via gr.Markdown inside accordions */
.note {
    background:  var(--blue-50);
    border:      1px solid var(--blue-100);
    border-left: 3px solid var(--blue-500);
    border-radius: var(--radius-sm);
    padding:     .55rem .8rem;
    font-size:   .84rem;
    color:       var(--blue-700);
    margin:      .4rem 0 .6rem;
    line-height: 1.55;
}
.note-warn {
    background:       var(--amber-100);
    border-color:     #fde68a;
    border-left-color: var(--amber-700);
    color:            var(--amber-700);
}

/* ── Scrollbar ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--gray-100); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gray-400); }

/* ── Global focus ring ─────────────────────────────────────────────────────── */
*:focus-visible { outline: 2px solid var(--blue-600) !important; outline-offset: 2px !important; }

/* ── Footer ────────────────────────────────────────────────────────────────── */
/* Hide default Gradio footer; we render our own custom footer below */
footer { display: none !important; }
.rag-footer {
    text-align:  center;
    font-family: 'IBM Plex Mono', monospace;
    font-size:   .7rem;
    color:       var(--gray-400);
    margin-top:  1.5rem;
    padding-top: .75rem;
    border-top:  1px solid var(--gray-200);
}
.rag-footer a {
    color:           var(--blue-600);
    text-decoration: none;
    font-weight:     600;
}
.rag-footer a:hover { text-decoration: underline; }
"""
