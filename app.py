import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from code_files import translate_text, analyze_text


def main():
    st.set_page_config(
        page_title="Advanced Multilingual News Bias Analyzer",
        layout="wide",
        page_icon="🌍"
    )

    st.title("🌍 Advanced Multilingual News Bias Analyzer")
    st.markdown(
        "### Upload articles → Analyze → Compare sentiment, emotions & bias across languages"
    )

    st.sidebar.header("📂 Upload News Articles")
    uploaded_files = st.sidebar.file_uploader(
        "Upload .txt files",
        type=["txt"],
        accept_multiple_files=True
    )

    st.sidebar.markdown("---")
    st.sidebar.info("Built with Streamlit, IBM NLU & Gemini Translation")

    if uploaded_files and st.sidebar.button("Run Full Analysis"):
        results = []
        full_texts = {}
        translations = {}

        st.header("📊 Analysis Dashboard")
        progress = st.progress(0.0)

        for idx, file in enumerate(uploaded_files):
            text = file.read().decode("utf-8", errors="ignore")

            if len(text) > 15000:
                st.warning(f"{file.name} is large. Truncating to 15,000 characters.")
                text = text[:15000]

            full_texts[file.name] = text

            with st.spinner(f"Translating {file.name}..."):
                try:
                    translated = translate_text(text)
                except Exception:
                    st.error(f"Translation failed for {file.name}. Using original text.")
                    translated = text

            translations[file.name] = translated

            with st.spinner(f"Analyzing {file.name}..."):
                try:
                    sentiment, emotions = analyze_text(translated)
                except Exception:
                    st.error(f"Analysis failed for {file.name}. Using default values.")
                    sentiment = 0.0
                    emotions = {
                        "joy": 0.0,
                        "sadness": 0.0,
                        "anger": 0.0,
                        "fear": 0.0,
                        "disgust": 0.0
                    }

            results.append({
                "Article": file.name,
                "Sentiment": sentiment,
                "joy": emotions.get("joy", 0.0),
                "sadness": emotions.get("sadness", 0.0),
                "anger": emotions.get("anger", 0.0),
                "fear": emotions.get("fear", 0.0),
                "disgust": emotions.get("disgust", 0.0)
            })

            progress.progress((idx + 1) / len(uploaded_files))

        df = pd.DataFrame(results)

        st.subheader("📌 Summary Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Articles Analyzed", len(df))
        col2.metric("Average Sentiment", round(df["Sentiment"].mean(), 3))

        dominant_emotion = (
            df.drop(columns=["Article", "Sentiment"])
              .mean()
              .idxmax()
              .capitalize()
        )
        col3.metric("Dominant Emotion", dominant_emotion)

        st.markdown("---")

        st.subheader("📈 Sentiment Comparison")
        fig_sent = px.bar(
            df,
            x="Article",
            y="Sentiment",
            color="Sentiment",
            color_continuous_scale="RdYlGn",
            title="Sentiment Score per Article"
        )
        st.plotly_chart(fig_sent, use_container_width=True)

        st.subheader("🔥 Emotion Heatmap")
        emotion_df = df.set_index("Article")[["joy", "sadness", "anger", "fear", "disgust"]]
        fig_heat = px.imshow(
            emotion_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("🕸 Emotion Radar Chart")
        categories = ["joy", "sadness", "anger", "fear", "disgust"]
        radar_fig = go.Figure()

        for _, row in df.iterrows():
            radar_fig.add_trace(go.Scatterpolar(
                r=[row[c] for c in categories],
                theta=categories,
                fill="toself",
                name=row["Article"]
            ))

        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True
        )
        st.plotly_chart(radar_fig, use_container_width=True)

        st.markdown("---")

        st.subheader("⬇ Download Results")
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name="analysis_results.csv"
        )

        st.header("📝 Detailed Article Analysis")
        selected_article = st.selectbox(
            "Choose an article to view details:",
            df["Article"]
        )

        colA, colB = st.columns(2)

        with colA:
            st.subheader("📄 Original Text")
            st.text_area(
                "Original",
                full_texts[selected_article],
                height=300
            )

        with colB:
            st.subheader("🌐 Translated Text (English)")
            st.text_area(
                "Translated",
                translations[selected_article],
                height=300
            )

        st.subheader("📊 Emotion Breakdown")
        emo_row = df[df["Article"] == selected_article].iloc[0]

        emo_data = pd.DataFrame({
            "Emotion": categories,
            "Score": [emo_row[c] for c in categories]
        })

        emo_fig = px.bar(
            emo_data,
            x="Emotion",
            y="Score",
            color="Emotion",
            title="Emotion Intensity"
        )
        st.plotly_chart(emo_fig, use_container_width=True)

    else:
        st.info("Upload text files from the sidebar and click **Run Full Analysis**.")


if __name__ == "__main__":
    main()
