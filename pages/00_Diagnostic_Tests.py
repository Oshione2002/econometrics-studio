from pathlib import Path

# Keep a top-positioned, clearly named entry in Streamlit's page navigation.
# The full implementation remains in 8_Diagnostics.py so both entry points
# share exactly the same diagnostics, export state, dark mode and clear logic.
implementation = Path(__file__).with_name("8_Diagnostics.py")
exec(compile(implementation.read_text(encoding="utf-8"), str(implementation), "exec"))
