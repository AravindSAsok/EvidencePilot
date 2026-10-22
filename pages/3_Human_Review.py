
import hashlib
import json
from datetime import datetime, timezone

import streamlit as st


VERDICTS = [
    "supported",
    "partially_supported",
    "unsupported",
    "unclear",
]

VERDICT_LABELS = {
    "supported": "Supported",
    "partially_supported": "Partially supported",
    "unsupported": "Unsupported",
    "unclear": "Unclear",
}

NOT_REVIEWED = "Not reviewed"


st.set_page_config(
    page_title="EvidencePilot | Human Review",
    page_icon="E",
    layout="wide",
)

from startup_screen import show_startup_screen

show_startup_screen("answer_audit")


st.markdown(
    """
    <style>
    :root { color-scheme: dark; }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #0B0F14 !important;
        color: #E8EDF2 !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .block-container {
        max-width: 1080px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }

    [data-testid="stSidebar"] {
        background: #101720 !important;
        border-right: 1px solid #2B3540 !important;
    }

    h1, h2, h3, h4 {
        color: #F0F3F6 !important;
        letter-spacing: -0.035em;
    }

    h1 {
        font-size: 2.25rem !important;
        font-weight: 600 !important;
    }

    p, li, label,
    [data-testid="stMarkdownContainer"] {
        color: #CAD2DC;
    }

    .ep-eyebrow {
        color: #A6C8B1 !important;
        font-size: 0.72rem;
        font-weight: 650;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .ep-muted {
        color: #A8B3C0 !important;
        font-size: 0.85rem;
        line-height: 1.65;
    }

    .ep-panel {
        background: #141B23 !important;
        color: #E8EDF2 !important;
        border: 1px solid #2B3540 !important;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin: 0.8rem 0 1.2rem 0;
    }

    .ep-divider {
        border: 0;
        border-top: 1px solid #2B3540;
        margin: 1.5rem 0;
    }

    div.stButton > button,
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stFormSubmitButton"] button {
        background: #A6C8B1 !important;
        color: #102018 !important;
        border: 1px solid #A6C8B1 !important;
        border-radius: 6px;
        min-height: 2.6rem;
        font-weight: 650;
    }

    div.stButton > button:hover,
    div[data-testid="stDownloadButton"] button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #C0DBC9 !important;
        border-color: #C0DBC9 !important;
        color: #102018 !important;
    }

    div[data-testid="stExpander"] {
        background: #141B23 !important;
        border: 1px solid #2B3540 !important;
        border-radius: 7px;
    }

    [data-testid="stMetric"] {
        background: #141B23 !important;
        border: 1px solid #2B3540;
        border-radius: 7px;
        padding: 0.8rem;
    }

    [data-testid="stMetricLabel"] {
        color: #A8B3C0 !important;
    }

    [data-testid="stMetricValue"] {
        color: #E8EDF2 !important;
    }

    footer { visibility: hidden; }

    @media (max-width: 700px) {
        .block-container { padding-top: 1.4rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_key_status():
    key = ""
    try:
        key = str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    if not key:
        import os
        key = os.getenv("GEMINI_API_KEY", "").strip()

    return bool(key)


def get_bundle_signature(bundle):
    payload = {
        "file_hash": bundle.get("file_hash"),
        "question": bundle.get("question"),
        "answer": bundle.get("answer"),
        "evidence": [
            {
                "page": item.get("page"),
                "text": item.get("text"),
            }
            for item in bundle.get("evidence", [])
        ],
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def calculate_agreement(human_labels, ai_labels):
    reviewed = [
        (human, ai)
        for human, ai in zip(human_labels, ai_labels)
        if human in VERDICTS and ai in VERDICTS
    ]

    count = len(reviewed)

    if not count:
        return {
            "reviewed_count": 0,
            "exact_matches": 0,
            "agreement_rate": None,
            "cohens_kappa": None,
            "confusion_matrix": None,
        }

    exact_matches = sum(
        human == ai for human, ai in reviewed
    )

    observed_agreement = exact_matches / count

    matrix = {
        human: {ai: 0 for ai in VERDICTS}
        for human in VERDICTS
    }

    for human, ai in reviewed:
        matrix[human][ai] += 1

    human_counts = {
        verdict: sum(matrix[verdict].values())
        for verdict in VERDICTS
    }

    ai_counts = {
        verdict: sum(matrix[human][verdict] for human in VERDICTS)
        for verdict in VERDICTS
    }

    expected_agreement = sum(
        human_counts[verdict] * ai_counts[verdict]
        for verdict in VERDICTS
    ) / (count * count)

    if abs(1 - expected_agreement) < 1e-12:
        kappa = None
    else:
        kappa = (
            observed_agreement - expected_agreement
        ) / (1 - expected_agreement)

    return {
        "reviewed_count": count,
        "exact_matches": exact_matches,
        "agreement_rate": observed_agreement,
        "cohens_kappa": kappa,
        "confusion_matrix": matrix,
    }


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="font-size:1.1rem;font-weight:650;color:#E8EDF2;">
            <span style="
                display:inline-flex;width:32px;height:32px;
                align-items:center;justify-content:center;
                border-radius:6px;background:#A6C8B1;color:#102018;
                margin-right:9px;font-size:0.8rem;font-weight:800;
            ">EP</span>EvidencePilot
        </div>
        <p class="ep-muted" style="margin-top:9px;">
            Human review workspace
        </p>
        <hr class="ep-divider">
        <div class="ep-eyebrow">Review guidance</div>
        <p class="ep-muted">
            Label claims using the retrieved evidence, not your prior
            assumptions or the AI evaluator's verdict.
        </p>
        <hr class="ep-divider">
        <p class="ep-muted">
            Your labels remain in the current app session unless you
            download the review report.
        </p>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="ep-eyebrow">QUALITY ASSURANCE</div>',
    unsafe_allow_html=True,
)

st.title("Human review lab")

st.markdown(
    """
    <p class="ep-muted">
        Independently review claims from an AI-audited answer, compare
        your verdicts with the evaluator, and measure their agreement.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr class='ep-divider'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Require an answer and completed AI audit from the current session
# ---------------------------------------------------------------------

bundle = st.session_state.get("answer_bundle")
saved_audit = st.session_state.get("answer_audit")

if not bundle or not bundle.get("answer") or not bundle.get("evidence"):
    st.info(
        "No completed research answer is available. Open the Evidence "
        "workspace, upload a PDF, and generate an answer first."
    )
    st.stop()

if bundle.get("file_hash") != st.session_state.get("pdf_hash"):
    st.warning(
        "The answer doesn't match the current PDF. Generate a fresh "
        "answer in the Evidence workspace."
    )
    st.stop()

if not saved_audit or not saved_audit.get("result"):
    st.info(
        "Run **Audit answer against retrieved evidence** on the Answer "
        "Audit page first. Then return here to label the claims."
    )
    st.stop()

signature = get_bundle_signature(bundle)

if saved_audit.get("signature") != signature:
    st.warning(
        "The audit no longer matches the latest answer. Re-run the "
        "Answer Audit, then return here."
    )
    st.stop()

audit = saved_audit["result"]
claims = audit.get("claims", [])

if not claims:
    st.info("The AI audit did not extract any claims to review.")
    st.stop()

review_store = st.session_state.setdefault("human_reviews", {})
saved_review = review_store.get(signature, {})

st.markdown("### Answer under review")

st.markdown(
    f"""
    <div class="ep-panel">
        <div style="font-weight:600;color:#E8EDF2;margin-bottom:0.6rem;">
            {bundle.get("question", "Research question")}
        </div>
        <div class="ep-muted">
            Retrieval method: {bundle.get("mode", "unknown")}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Retrieved evidence")

st.markdown(
    """
    <p class="ep-muted">
        Read these passages before labelling claims. The claim-level AI
        verdicts are intentionally hidden while you perform the review.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.expander(
    f"Inspect source passages ({len(bundle['evidence'])})",
    expanded=True,
):
    for index, item in enumerate(bundle["evidence"], start=1):
        st.markdown(f"**Evidence E{index} · Page {item.get('page', '?')}**")
        st.write(item.get("text", ""))
        st.markdown("<hr class='ep-divider'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Human labels
# ---------------------------------------------------------------------

st.markdown("<hr class='ep-divider'>", unsafe_allow_html=True)
st.markdown("### Independently label each claim")

st.caption(
    "Supported: the evidence substantively establishes the claim. "
    "Partially supported: only some of it is established. "
    "Unsupported: the evidence does not establish it. "
    "Unclear: more context is needed."
)

widget_prefix = signature[:12]

with st.form("human_review_form"):
    current_labels = {}

    for index, claim in enumerate(claims):
        key = str(index)
        existing = saved_review.get(key, NOT_REVIEWED)

        if existing not in [NOT_REVIEWED, *VERDICTS]:
            existing = NOT_REVIEWED

        st.markdown(f"**Claim {index + 1}**")
        st.write(claim.get("claim", ""))

        citation_pages = claim.get("cited_pages", [])

        st.caption(
            "Pages cited in the answer: "
            + (", ".join(map(str, citation_pages)) if citation_pages else "None")
        )

        options = [NOT_REVIEWED, *VERDICTS]
        default_index = options.index(existing)

        current_labels[key] = st.selectbox(
            f"Your verdict for claim {index + 1}",
            options=options,
            index=default_index,
            format_func=lambda value: (
                value.replace("_", " ").title()
            ),
            key=f"human_label_{widget_prefix}_{index}",
        )

        st.markdown("<hr class='ep-divider'>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "Save human review",
        type="primary",
    )

if submitted:
    updated_labels = {
        str(index): st.session_state[
            f"human_label_{widget_prefix}_{index}"
        ]
        for index in range(len(claims))
    }

    review_store[signature] = updated_labels
    st.session_state["human_reviews"] = review_store

    st.success("Human review saved in the current session.")
    st.rerun()


# ---------------------------------------------------------------------
# Compare human labels against AI verdicts
# ---------------------------------------------------------------------

saved_review = st.session_state.get("human_reviews", {}).get(signature, {})

human_labels = [
    saved_review.get(str(index), NOT_REVIEWED)
    for index in range(len(claims))
]

ai_labels = [
    claim.get("verdict", "unclear")
    for claim in claims
]

reviewed_count = sum(
    label in VERDICTS for label in human_labels
)

st.markdown("<hr class='ep-divider'>", unsafe_allow_html=True)
st.markdown(
    '<div class="ep-eyebrow">COMPARISON REPORT</div>',
    unsafe_allow_html=True,
)

st.markdown("### Human vs AI evaluator")

st.caption(
    f"{reviewed_count} of {len(claims)} claims have a saved human label."
)

if reviewed_count == 0:
    st.info("Label and save at least one claim to calculate agreement.")
    st.stop()

reviewed_human = [
    human_labels[index]
    for index in range(len(claims))
    if human_labels[index] in VERDICTS
]

reviewed_ai = [
    ai_labels[index]
    for index in range(len(claims))
    if human_labels[index] in VERDICTS
]

agreement = calculate_agreement(reviewed_human, reviewed_ai)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Claims reviewed",
        agreement["reviewed_count"],
        f"of {len(claims)}",
    )

with col2:
    match_rate = agreement["agreement_rate"]

    st.metric(
        "Exact agreement",
        f"{match_rate:.1%}" if match_rate is not None else "N/A",
        f"{agreement['exact_matches']} matching verdicts",
    )

with col3:
    kappa = agreement["cohens_kappa"]

    st.metric(
        "Cohen's kappa",
        f"{kappa:.3f}" if kappa is not None else "N/A",
    )

st.caption(
    "Exact agreement is the percentage of reviewed claims where the human "
    "and AI selected the same verdict. Cohen's kappa adjusts agreement for "
    "chance agreement. A value may be undefined when the label distribution "
    "contains no variation."
)

if agreement["reviewed_count"] < 10:
    st.warning(
        "This is a very small sample. Treat the agreement metrics as "
        "exploratory, not as evidence of evaluator reliability."
    )


# Confusion matrix: rows are human labels, columns are AI labels
matrix = agreement["confusion_matrix"]

if matrix:
    st.markdown("#### Confusion matrix")
    st.caption("Rows = human verdicts · Columns = AI evaluator verdicts")

    matrix_rows = []

    for human_verdict in VERDICTS:
        row = {
            "Human verdict": VERDICT_LABELS[human_verdict],
        }

        for ai_verdict in VERDICTS:
            row[VERDICT_LABELS[ai_verdict]] = matrix[human_verdict][ai_verdict]

        matrix_rows.append(row)

    st.dataframe(
        matrix_rows,
        use_container_width=True,
        hide_index=True,
    )


st.markdown("#### Per-claim comparison")

comparison_rows = []

for index, claim in enumerate(claims):
    human = human_labels[index]
    ai = ai_labels[index]

    comparison_rows.append({
        "Claim": claim.get("claim", ""),
        "Human verdict": (
            VERDICT_LABELS[human] if human in VERDICTS else NOT_REVIEWED
        ),
        "AI verdict": VERDICT_LABELS.get(ai, ai),
        "Match": (
            human == ai if human in VERDICTS else None
        ),
    })

st.dataframe(
    comparison_rows,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------------------
# Export the review report
# ---------------------------------------------------------------------

report = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_filename": st.session_state.get("pdf_name"),
    "source_sha256": bundle.get("file_hash"),
    "question": bundle.get("question"),
    "answer": bundle.get("answer"),
    "retrieval_method": bundle.get("mode"),
    "claim_reviews": [
        {
            "claim": claims[index].get("claim", ""),
            "human_verdict": (
                human_labels[index]
                if human_labels[index] in VERDICTS
                else None
            ),
            "ai_verdict": ai_labels[index],
            "human_ai_match": (
                human_labels[index] == ai_labels[index]
                if human_labels[index] in VERDICTS
                else None
            ),
            "ai_explanation": claims[index].get("explanation", ""),
            "supporting_passage_ids": claims[index].get(
                "supporting_passage_ids", []
            ),
        }
        for index in range(len(claims))
    ],
    "metrics": {
        "reviewed_count": agreement["reviewed_count"],
        "total_claims": len(claims),
        "exact_matches": agreement["exact_matches"],
        "agreement_rate": agreement["agreement_rate"],
        "cohens_kappa": agreement["cohens_kappa"],
        "confusion_matrix": agreement["confusion_matrix"],
    },
    "limitations": [
        "Human labels may also contain errors.",
        "One reviewer cannot establish inter-annotator reliability.",
        "A small number of claims is insufficient for strong statistical conclusions.",
        "AI verdict agreement does not prove factual correctness.",
    ],
}

st.download_button(
    "Download human review report",
    data=json.dumps(report, indent=2, ensure_ascii=False),
    file_name="evidencepilot-human-review.json",
    mime="application/json",
)

st.markdown("<hr class='ep-divider'>", unsafe_allow_html=True)
st.caption(
    "EvidencePilot · Human Review Lab · Review labels are session-based "
    "until you download the report."
)
