import streamlit as st
from allusion import AllusionAgent, save_report

LOGO_PATH = "assets/allusion_logo.png"

st.set_page_config(
    page_title="Allusion",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(173, 205, 142, 0.18), transparent 35%),
            radial-gradient(circle at top right, rgba(212, 184, 121, 0.14), transparent 30%),
            linear-gradient(135deg, #061512 0%, #0B1F19 45%, #11130F 100%);
        color: #F2EBCB;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071712 0%, #10251D 100%);
        border-right: 1px solid rgba(212, 184, 121, 0.25);
    }

    h1, h2, h3 {
        color: #F5EBC7;
        letter-spacing: 0.03em;
    }

    .hero-card {
    padding: 2rem;
    border-radius: 24px;
    background: rgba(10, 31, 25, 0.72);
    border: 1px solid rgba(212, 184, 121, 0.35);

    box-shadow:
        0 0 20px rgba(173, 205, 142, 0.12),
        0 0 50px rgba(173, 205, 142, 0.08),
        0 0 90px rgba(212, 184, 121, 0.05);
}

    .hero-title {
        font-size: 3.2rem;
        font-weight: 700;
        letter-spacing: 0.22em;
        color: #F4EBC9;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        color: #B6D28D;
        font-size: 1.05rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .hero-text {
        color: #D8CFAC;
        font-size: 1.05rem;
        margin-top: 1rem;
        line-height: 1.7;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(17, 44, 34, 0.65);
        border: 1px solid rgba(173, 205, 142, 0.25);
    }

    .stButton > button {
        background: linear-gradient(135deg, #B6D28D 0%, #D4B879 100%);
        color: #071712;
        border: none;
        border-radius: 14px;
        font-weight: 700;
        padding: 0.7rem 1.4rem;
    }

    .stTextInput input {
    background-color: rgba(8, 25, 20, 0.95);
    color: #F5EBC7;
    border: 1px solid rgba(212, 184, 121, 0.55);
    border-radius: 14px;

    box-shadow:
        0 0 10px rgba(182, 210, 141, 0.15),
        0 0 20px rgba(182, 210, 141, 0.10),
        0 0 40px rgba(212, 184, 121, 0.08);

    transition: all 0.3s ease;
}

.stTextInput input:focus {
    border: 1px solid rgba(182, 210, 141, 0.9);

    box-shadow:
        0 0 15px rgba(182, 210, 141, 0.35),
        0 0 35px rgba(182, 210, 141, 0.20),
        0 0 70px rgba(212, 184, 121, 0.15);
}

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(17, 44, 34, 0.7);
        border-radius: 12px;
        color: #D8CFAC;
        padding: 10px 18px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(182, 210, 141, 0.35), rgba(212, 184, 121, 0.28));
        color: #F5EBC7;
    }

    a {
        color: #B6D28D !important;
    }

    .small-muted {
        color: #9CA98B;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.image(LOGO_PATH, width=420)

with col_title:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">ALLUSION</div>
            <div class="hero-subtitle">Research Scout for Emerging Ideas</div>
            <div class="hero-text">
                Discover public sources, trace weak signals, and trailblaze your projects
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("Search Settings")

    max_results = st.slider(
        "Max results",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
    )

    delay = st.slider(
        "Delay between fetches",
        min_value=0.0,
        max_value=3.0,
        value=0.75,
        step=0.25,
    )

    include_social = st.checkbox(
        "Include social/video sources",
        value=False,
    )

    verbose = st.checkbox(
        "Research Debug Mode",
        value=False,
    )

st.markdown("## Discover Hidden Signals")

st.caption("Emerging Ideas • Hidden Patterns • Project Inspiration")

topic = st.text_input(
    "Research topic",
    placeholder='Example: "underexplored AI agent architectures"',
)

run = st.button("Run Allusion", type="primary")

if run:
    if not topic.strip():
        st.warning("Enter a research topic first.")
    else:
        with st.spinner("Exploring public sources..."):
            agent = AllusionAgent(
                delay_seconds=delay,
                include_social=include_social,
                verbose=verbose,
            )

            report = agent.explore(
                topic.strip(),
                max_results=max_results,
            )

            path = save_report(report, topic.strip())

        st.success(f"Report saved to: {path}")

        st.markdown(
            f"""
            <div class="metric-card">
                <strong>Sources collected:</strong> {len(report.sources)} &nbsp; | &nbsp;
                <strong>Niche signals:</strong> {len(report.niche_signals)} &nbsp; | &nbsp;
                <strong>Themes:</strong> {len(report.themes)} &nbsp; | &nbsp;
                <strong>Allusions:</strong> {len(report.allusions)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Report",
                "Signal Map",
                "Ranked Sources",
                "Raw Markdown",
            ]
        )

        with tab1:
            st.markdown(report.markdown)

        with tab2:
            st.subheader("What Looks Niche")
            for signal in report.niche_signals:
                st.write(f"- {signal}")

            st.divider()

            st.subheader("Repeated Themes")
            for theme in report.themes:
                st.write(f"- {theme}")

            st.divider()

            st.subheader("Allusions in the Machine")
            for item in report.allusions:
                st.write(f"- {item}")

        with tab3:
            ranked = sorted(
                report.sources,
                key=lambda d: d.niche_score,
                reverse=True,
            )

            for doc in ranked:
                st.markdown(f"### [{doc.title}]({doc.url})")
                st.write(f"**Domain:** {doc.domain}")
                st.write(f"**Status:** {doc.status}")
                st.write(f"**Niche Score:** {doc.niche_score:.0f}/100")
                st.write(f"**Mainstream Score:** {doc.mainstream_score:.0f}/100")

                if doc.keywords:
                    st.write("**Keywords:** " + ", ".join(doc.keywords[:8]))

                if doc.niche_reasons:
                    st.write(f"**Why it matters:** {doc.niche_reasons[0]}")

                st.divider()

        with tab4:
            st.code(report.markdown, language="markdown")

            st.download_button(
                label="Download Markdown Report",
                data=report.markdown,
                file_name="allusion_report.md",
                mime="text/markdown",
            )
