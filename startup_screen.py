import time
import streamlit as st


def show_startup_screen(page_key: str, duration: int = 3) -> None:
    """Show a full-viewport branded loading overlay once per page per session."""
    state_key = f"ep_startup_screen_shown_{page_key}"
    if st.session_state.get(state_key, False):
        return

    overlay = st.empty()
    overlay.markdown(
        f"""
        <div style="position:fixed;inset:0;z-index:2147483647;background:#0B0F14;display:flex;align-items:center;justify-content:center;text-align:center;padding:24px;">
          <div style="max-width:680px;width:100%;">
            <div style="width:66px;height:66px;margin:0 auto 24px;border:1px solid #52665A;border-radius:18px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#1B2B22,#111922);color:#A6C8B1;font-size:27px;font-weight:800;letter-spacing:-2px;box-shadow:0 0 50px rgba(166,200,177,.10);">EP</div>
            <div style="color:#A6C8B1;font-size:12px;font-weight:750;letter-spacing:3px;text-transform:uppercase;margin-bottom:14px;">YOUR RESEARCH WORKSPACE</div>
            <div style="color:#F0F3F6;font-size:clamp(34px,5vw,54px);font-weight:750;letter-spacing:-2px;line-height:1.08;margin-bottom:16px;">EvidencePilot</div>
            <div style="color:#DCE9E0;font-size:19px;margin-bottom:12px;">Research, Grounded in Evidence.</div>
            <div style="color:#A8B3C0;font-size:14px;line-height:1.8;margin-bottom:30px;">Explore documents. Investigate claims. Trace every insight back to its source.</div>
            <div style="display:inline-flex;align-items:center;gap:10px;border:1px solid #35483B;background:#121D17;border-radius:999px;padding:10px 16px;color:#DCE9E0;font-size:12px;">
              <span style="width:7px;height:7px;border-radius:50%;background:#A6C8B1;display:inline-block;box-shadow:0 0 12px #A6C8B1;"></span>
              Preparing your workspace…
            </div>
            <div style="height:3px;width:190px;background:#26312B;border-radius:9px;margin:22px auto 0;overflow:hidden;"><div style="height:100%;width:100%;background:linear-gradient(90deg,#678B73,#C0D9C6);transform-origin:left;animation:ep-startup-progress {duration}s linear forwards;"></div></div>
          </div>
        </div>
        <style>@keyframes ep-startup-progress {{ from {{ transform:scaleX(0); }} to {{ transform:scaleX(1); }} }}</style>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(duration)
    overlay.empty()
    st.session_state[state_key] = True
