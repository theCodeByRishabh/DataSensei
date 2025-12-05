import streamlit as st
from streamlit_lottie import st_lottie
import requests
import json
from PIL import Image



st.set_page_config(page_title="DataSensei", page_icon=":😎:", layout="wide",initial_sidebar_state="collapsed")
with open('header.html', 'r') as file:
        header_html_content = file.read()
st.markdown(header_html_content, unsafe_allow_html=True)


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
lottie_student=load_lottieurl("https://lottie.host/f658798b-95d9-4ca5-a1fc-6bee7ad9e4ba/MQAFnVrvkD.json")
lottie_python=load_lottieurl("https://lottie.host/9e7454b3-afde-4773-8fbc-cee0f6cfc181/kp0uyQ9xV1.json")
lottie_clean=load_lottieurl("https://lottie.host/7b0d9a9c-c40a-4063-9708-0d2960752c82/pKDZHJjdx9.json")
lottie_analyze=load_lottieurl("https://lottie.host/0c85bc6e-f558-4037-828a-c5755c477f70/rnNFjmw8BI.json")
lottie_visualize=load_lottieurl("https://lottie.host/da36a285-789c-4684-b42a-2a81769b50dc/ZDgBsLgdSX.json")
lottie_tableau=load_lottieurl("https://lottie.host/41af9409-16f8-4822-888c-cb92e13c6f05/jHhf6LCpkJ.json")
lottie_community=load_lottieurl("https://lottie.host/b1517e1e-6959-4c59-8775-beda85c45e31/8sTim1a9u1.json")
lottie_chatbot=load_lottieurl("https://lottie.host/7bd3be4a-6e0f-48da-8b2e-a20de793c1e2/IxVWb7LC7A.json")
lottie_courses=load_lottieurl("https://lottie.host/397395a0-d7c6-484e-be79-10565eaa2f8c/5MdKgAbL3O.json")



introduction="""
Welcome to DataSensei, where the world of data becomes accessible, insightful, and exciting. 
Our platform is designed to empower individuals from all backgrounds, whether you're a budding data analyst, an experienced data scientist, or simply someone intrigued by the power of data. 
At DataSensei, we offer an array of tools and services that streamline your data journey, including robust data cleaning, in-depth analytics, captivating visualizations, and even real-time interactions with our intuitive chatbot. 
Our mission is to simplify the complexities of working with data, making it a delightful and engaging experience. 
Join us on this transformative adventure and unleash the true potential of your data. Let's get started!"
"""

about="""
Discover the limitless potential of your data with our innovative platform, DataSensei. 
Seamlessly upload and clean datasets with our data cleaning tools, ensuring accuracy and reliability. 
Dive into comprehensive data analytics, including hypothesis testing, regression analysis, and clustering. 
Transform raw data into captivating visual stories using a wide array of graphs and plots. 
Engage with our interactive chatbot and join a vibrant community of data enthusiasts. 
DataSensei empowers you to explore, analyze, and communicate data-driven insights effortlessly
"""

technology="""
DataSensei brings together a powerhouse of technologies to enhance your data journey. 
Utilizing Python libraries like NumPy and Pandas, we ensure seamless data manipulation and cleaning. 
Our platform employs Matplotlib and Seaborn to visualize insights, while PygWalker crafts a web-based Tableau experience. 
With a foundation built on Streamlit, HTML, CSS, and JavaScript, DataSensei delivers a dynamic and engaging interface, enabling you to unlock the true potential of your data through efficient cleaning, in-depth analysis, and captivating visualization.
"""

cleaning="""
Elevate your data quality with DataSensei's robust Data Cleaning. 
Seamlessly upload CSV or XLSX files and let our tools work magic. 
We'll handle missing values, correct data types, remove duplicates, and more. 
Text data receives special attention with tokenization, lowercasing, and stemming. 
The module's intuitive interface guides you through the process, ensuring clean, accurate, and reliable data ready for analysis.
"""

analytics="""
Uncover valuable insights with DataSensei's Data Analytics. 
Upload your data files and unlock a world of analysis possibilities. 
From hypothesis testing and regression analysis to time series examination and powerful clustering techniques, our platform equips you with advanced tools. 
Python's libraries drive our analytics engine, guiding you through complex data patterns and trends. 
With Data Analytics, you'll make informed decisions and extract meaningful knowledge from your data effortlessly.
"""

visualize="""
Transform data into captivating stories through DataSensei's Data Visualization. 
Upload your datasets and explore a rich array of visualization options. 
From classic line and bar plots to intricate 3D scatter plots and word clouds, our platform offers versatile graphical representations. 
Choose the desired graph type, select relevant columns, and watch your data come to life. 
With Matplotlib, Seaborn, and interactive elements, Data Visualization empowers you to present insights in engaging, informative ways.
"""

tableau="""
Experience the power of interactive data exploration with DataSensei's Tableau. 
Upload your CSV or XLSX files and immerse yourself in a web-based Tableau-like environment. 
PygWalker technology drives this intuitive interface, allowing you to visualize, analyze, and interact with your data seamlessly. 
Explore charts, maps, and dashboards with ease, without the need for local installations. 
DataSensei's Tableau module empowers you to uncover insights, spot trends, and make data-driven decisions effortlessly
"""

community="""
Engage and collaborate in a vibrant data-driven community through DataSensei's Community. 
Connect with fellow data enthusiasts, ask questions, share insights, and exchange knowledge effortlessly. 
Whether you're a beginner seeking guidance or an expert offering valuable advice, our platform fosters open discussions and collaborative learning. 
Join discussions, post queries, and contribute to a supportive ecosystem dedicated to enhancing data understanding. 
With DataSensei's Community module, you're part of a dynamic space where insights and experiences converge for mutual growth.
"""

chat="""
Elevate your interactions with data using DataSensei's ChatBot. 
Engage in real-time conversations with our intelligent virtual assistant that responds to both text and speech input. 
Powered by speech recognition and Wikipedia integration, the ChatBot provides quick access to information and answers. 
Whether you're curious about data concepts or seeking specific insights, the ChatBot delivers accurate responses. 
With a seamless blend of technology and communication, DataSensei's ChatBot enhances your data learning experience and provides instant assistance.
"""


courses="""
Empower your data journey with DataSensei's Courses. 
Unlock comprehensive learning resources tailored for aspiring data analysts and scientists. 
Choose from a variety of text-based or video-based courses, designed to equip you with essential skills and knowledge. 
Our courses cover a wide range of topics, from data fundamentals to advanced analytics techniques. 
Whether you're looking to upskill or start from scratch, DataSensei's Courses module provides a structured path to mastering the art of data analysis and visualization.
"""


const="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc at ligula auctor, ultrices ipsum ut, tincidunt dui. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Nunc non sem id leo sollicitudin luctus nec eu sem. Cras vitae neque nec lorem ullamcorper pretium. In congue et orci non varius. Sed convallis arcu velit, id pellentesque quam eleifend blandit. Nullam arcu neque, scelerisque nec ornare in, sodales eget est. Donec sagittis tellus libero. Nunc magna lorem, pharetra iaculis mattis eu, consequat vitae urna."

name='User'
popup_slot = st.empty()

with st.container():
    left_column, right_column = st.columns(2)
    with right_column:
        st.write("Hi :wave:" , name)
        st.title("Introduction")
        st.write(introduction)
    with left_column:
        st_lottie(lottie_student, height=300, key="student",quality="high")



st.markdown(
    """
<br>
""",unsafe_allow_html=True
)


st.write("---")


st.markdown(
    """
<br>
""",unsafe_allow_html=True
)


with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown("""<br><br>""",unsafe_allow_html=True)
        st.title("About the Website")
        st.write(about)
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding",quality="high")



st.markdown(
    """
<br>
""",unsafe_allow_html=True
)


st.write("---")


st.markdown(
    """
<br>
""",unsafe_allow_html=True
)

with st.container():
    left_column, right_column = st.columns(2)
    with right_column:
        st.title("Technologies")
        st.write(technology)
    with left_column:
        st_lottie(lottie_python, height=300, key="python",quality="high")



st.markdown(
    """
<br>
""",unsafe_allow_html=True
)


st.write("---")


st.markdown(
    """
<br>
""",unsafe_allow_html=True
)

st.markdown(
    """
<h2 style="text-align:center;">The Three D's</h2>

""",unsafe_allow_html=True
)

st.write("---")

st.markdown(
    """
<br>
""",unsafe_allow_html=True
)


with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Data Cleaning")
        st.write(cleaning)
    with right_column:
        st_lottie(lottie_clean, height=300, key="clean",quality="high")



st.markdown(
    """
<br>
""",unsafe_allow_html=True
)


st.write("---")


st.markdown(
    """
<br>
""",unsafe_allow_html=True
)



with st.container():
    left_column, right_column = st.columns(2)
    with right_column:
        st.title("Data Analytics")
        st.write(analytics)
    with left_column:
        st_lottie(lottie_analyze, height=300, key="analyze",quality="high")


st.markdown(
    """
<br>
""",unsafe_allow_html=True
)

st.write("---")

st.markdown(
    """
<br>
""",unsafe_allow_html=True
)


with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Data Visualization")
        st.write(visualize)
    with right_column:
        st_lottie(lottie_visualize, height=300, key="visualize",quality="high")





st.markdown(
    """
<br>
""",unsafe_allow_html=True
)


st.write("---")


st.markdown(
    """
<br>
""",unsafe_allow_html=True
)

st.markdown(
    """
<h2 style="text-align:center;">Our Interactive Tools & Services</h2>

""",unsafe_allow_html=True
)

st.write("---")

st.markdown(
    """
<br>
""",unsafe_allow_html=True
)






with st.container():
    left_column, right_column = st.columns(2)
    with right_column:
        st.title("Tableau")
        st.write(tableau)
    with left_column:
        st_lottie(lottie_tableau, height=300, key="tableau",quality="high")


st.markdown(
    """
<br>
""",unsafe_allow_html=True
)

st.write("---")

st.markdown(
    """
<br>
""",unsafe_allow_html=True)






with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Community")
        st.write(community)
    with right_column:
        st_lottie(lottie_community, height=300, key="community",quality="high")


st.markdown(
    """
<br>
""",unsafe_allow_html=True
)

st.write("---")

st.markdown(
    """
<br>
""",unsafe_allow_html=True)

st.write("---")





