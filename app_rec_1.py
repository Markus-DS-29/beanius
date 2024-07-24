from streamlit_mic_recorder import mic_recorder
audio = mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False,
    use_container_width=False,
    format="webm",
    callback=None,
    args=(),
    kwargs={},
    key=None
)
