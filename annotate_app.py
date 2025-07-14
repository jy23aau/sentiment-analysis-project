import streamlit as st
import pandas as pd
import os

SAMPLES_TO_ANNOTATE_FILE = 'samples_to_annotate.csv'
NEWLY_LABELED_DATA_FILE = 'newly_labeled_data.csv'
POSSIBLE_LABELS = [-3, -2, -1, 0, 1, 2, 3]

st.set_page_config(layout="wide", page_title="Sentiment Annotation Tool")

st.title("Sentiment Annotation Tool")
st.write("Label the sentiment of the student feedback sentences.")

if 'data' not in st.session_state:
    if os.path.exists(SAMPLES_TO_ANNOTATE_FILE):
        st.session_state.data = pd.read_csv(SAMPLES_TO_ANNOTATE_FILE)
        st.session_state.annotations = []
        st.session_state.current_index = 0
        st.success(f"Loaded {len(st.session_state.data)} samples to annotate.")
    else:
        st.error(f"File '{SAMPLES_TO_ANNOTATE_FILE}' not found. Run active learning script first.")
        st.stop()

if st.session_state.current_index < len(st.session_state.data):
    sample = st.session_state.data.iloc[st.session_state.current_index]
    text = sample['text']
    idx = sample.name

    st.subheader(f"Sample {st.session_state.current_index + 1} of {len(st.session_state.data)}")
    st.markdown(f"**Text:** _{text}_")

    selected_label = st.radio(
        "Select Label:",
        options=POSSIBLE_LABELS,
        index=POSSIBLE_LABELS.index(0),
        horizontal=True,
        key=f"label_radio_{st.session_state.current_index}"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit & Next", key="submit_next"):
            st.session_state.annotations.append({
                'original_unlabeled_index': idx,
                'text': text,
                'label': selected_label
            })
            st.session_state.current_index += 1
            st.rerun()
    with col2:
        if st.button("Skip", key="skip"):
            st.session_state.current_index += 1
            st.rerun()
else:
    st.success("All samples annotated!")
    if st.session_state.annotations:
        df = pd.DataFrame(st.session_state.annotations)
        if os.path.exists(NEWLY_LABELED_DATA_FILE):
            df.to_csv(NEWLY_LABELED_DATA_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(NEWLY_LABELED_DATA_FILE, index=False)
        st.download_button("Download Annotations CSV", data=df.to_csv(index=False).encode('utf-8'), file_name='annotations.csv')
        st.info(f"Annotations saved to '{NEWLY_LABELED_DATA_FILE}'.")
    else:
        st.info("No annotations made.")

    if st.button("Start Over"):
        if os.path.exists(SAMPLES_TO_ANNOTATE_FILE):
            st.session_state.data = pd.read_csv(SAMPLES_TO_ANNOTATE_FILE)
            st.session_state.current_index = 0
            st.session_state.annotations = []
            st.rerun()
        else:
            st.error(f"File '{SAMPLES_TO_ANNOTATE_FILE}' not found.")
