"""Look and feel. One injected stylesheet: cards, badges, notices, responsive grids, restrained industrial palette."""
from __future__ import annotations

import streamlit as st

CSS = """
<style>
:root { --pm-ink:#1E293B; --pm-muted:#64748B; --pm-line:#E2E8F0; --pm-card:#FFFFFF; --pm-bg:#F8FAFC; --pm-accent:#0F172A; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; color: var(--pm-ink); }
.pm-hero { background: linear-gradient(120deg,#0F172A 0%,#1E3A5F 60%,#0F766E 130%); color:#fff; border-radius:14px; padding:22px 26px; margin-bottom:14px; }
.pm-hero h1 { color:#fff; margin:0 0 4px 0; font-size:1.7rem; }
.pm-hero p { margin:0; color:#CBD5E1; font-size:0.95rem; }
.pm-badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.70rem; font-weight:700; letter-spacing:.06em; color:#fff; vertical-align:middle; }
.pm-card { background:var(--pm-card); border:1px solid var(--pm-line); border-radius:12px; padding:14px 16px; height:100%; }
.pm-card .lbl { font-size:0.72rem; text-transform:uppercase; letter-spacing:.07em; color:var(--pm-muted); font-weight:600; }
.pm-card .val { font-size:1.65rem; font-weight:700; color:var(--pm-ink); line-height:1.25; }
.pm-card .sub { font-size:0.80rem; color:var(--pm-muted); }
.pm-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin:6px 0 14px 0; }
.pm-notice { border-left:5px solid var(--c,#64748B); background:rgba(100,116,139,0.08); border-radius:6px; padding:10px 14px; margin:8px 0; font-size:0.88rem; color:var(--pm-ink); }
.pm-notice b { color:var(--c,#334155); }
.pm-chip { display:inline-block; padding:2px 10px; border-radius:6px; font-size:0.78rem; font-weight:600; color:#fff; }
.pm-section { display:flex; align-items:center; gap:10px; margin:6px 0 2px 0; flex-wrap:wrap; }
.pm-section .t { font-weight:700; font-size:1.05rem; color:var(--pm-ink); }
.pm-foot { color:var(--pm-muted); font-size:0.78rem; border-top:1px solid var(--pm-line); margin-top:26px; padding-top:10px; }
section[data-testid="stSidebar"] { border-right:1px solid var(--pm-line); }
@media (max-width: 640px) {
  .block-container { padding-left: 0.9rem; padding-right: 0.9rem; padding-top: 1.2rem; }
  .pm-hero { padding:16px 16px; } .pm-hero h1 { font-size:1.3rem; }
  .pm-card .val { font-size:1.3rem; }
}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
