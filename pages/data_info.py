import streamlit as st

Activities = ['Away','Bend','Crawl','Kneel','Limp','Pick','Scissor','Sit','SStep','Toes','Towards']
Radars = ['24GHz','77GHz','Xethru']

st.title("Human activity dataset from CI4R")

st.subheader("Descrption of the dataset")
st.markdown("Six participants of various ages, heights and weights were involved in this study. Three different sensors and a total of 11 diﬀerent activities. Each participant conducted 10 repetitions of each activity, resulting in a total of 60 samples per class per sensor.")

with st.expander("ACTIVITIES in the dataset"):
    st.markdown("""
    - **Away**: Walking away from the radar
    - **Bend**: Bending down
    - **Crawl**: Crawling on the floor
    - **Kneel**: Kneeling down
    - **Limp**: Walking with a limp
    - **Pick**: Picking up an object from the ground
    - **Scissor**: Scissor walking
    - **Sit**: Sitting down
    - **SStep**: Side stepping
    - **Toes**: Standing on toes
    - **Towards**: Walking towards the radar
    """)


st.subheader("Radar Sample Data Images")
for radar in Radars:
    with st.expander(f"***{radar}***"):
        cols = st.columns(4)
        for i, act in enumerate(Activities):
            img = f"data/{radar}/{act}/1.png"
            with cols[i % 4]:
                st.image(img, caption=act, use_column_width=True)