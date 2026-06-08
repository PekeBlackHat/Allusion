import streamlit as st
from allusion import AllusionAgent, save_report

st.set_page_config(
    page_title="Allusion",
    page_icon="🔎",
    layout="wide",
)

st.title("Allusion")
st.caption(
    "Discover emerging ideas, hidden patterns, and underexplored research signals."
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
        "Verbose mode",
        value=False,
    )

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

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Report",
                "Observations",
                "Ranked Sources",
                "Raw Markdown",
            ]
        )

        with tab1:
            st.markdown(report.markdown)

        with tab2:
            for obs in report.observations:
                st.subheader(obs.title)
                st.write(f"**Category:** {obs.category}")
                st.write(f"**Confidence:** {obs.confidence:.2f}")
                st.write(f"**Insight:** {obs.insight}")

                if obs.evidence:
                    st.write("**Evidence:**")
                    for item in obs.evidence:
                        st.write(f"- {item}")

        st.divider()

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
