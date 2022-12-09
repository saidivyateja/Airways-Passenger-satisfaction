import streamlit as st
import seaborn as sns
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
from sklearn import metrics
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score
# from st_aggrid import AgGrid, GridUpdateMode, JsCode
# from st_aggrid.grid_options_builder import GridOptionsBuilder

df1=pd.read_csv("train.csv")
df1=df1.dropna()
df=df1.drop(columns=['Unnamed: 0','id']).head(2500)
df['Gender'].replace(['Male', 'Female'],[0, 1], inplace=True)
df['Customer Type'].replace(['Loyal Customer', 'disloyal Customer'],[0, 1], inplace=True)
df['Type of Travel'].replace(['Personal Travel', 'Business travel'],[0, 1], inplace=True)
df['Class'].replace(['Eco Plus', 'Business','Eco'],[0, 1, 2], inplace=True)
df['satisfaction'].replace(['neutral or dissatisfied','satisfied'],[0,1],inplace=True)

st.title('Project on Airways Passenger Satisfaction')

def page1():

    st.markdown('''#### Presenter Name: Sai Divya Teja Konda  
        student id: 181243326''')
    
    image = Image.open('p1.png')

    st.image(image,width=500)

    st.header("Contents of the Dataset")

    st.markdown('''
    Gender: Gender of the passengers (Female, Male)

    Customer Type: The customer type (Loyal customer, disloyal customer)

    Age: The actual age of the passengers

    Type of Travel: Purpose of the flight of the passengers (Personal Travel, Business Travel)

    Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus)

    Flight distance: The flight distance of this journey

    Inflight wifi service: Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)

    Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient

    Ease of Online booking: Satisfaction level of online booking

    Gate location: Satisfaction level of Gate location

    Food and drink: Satisfaction level of Food and drink

    Online boarding: Satisfaction level of online boarding

    Seat comfort: Satisfaction level of Seat comfort

    Inflight entertainment: Satisfaction level of inflight entertainment

    On-board service: Satisfaction level of On-board service

    Leg room service: Satisfaction level of Leg room service

    Baggage handling: Satisfaction level of baggage handling

    Check-in service: Satisfaction level of Check-in service

    Inflight service: Satisfaction level of inflight service

    Cleanliness: Satisfaction level of Cleanliness

    Departure Delay in Minutes: Minutes delayed when departure

    Arrival Delay in Minutes: Minutes delayed when Arrival

    Satisfaction: Airline satisfaction level(Satisfaction, neutral or dissatisfaction)
        ''')

def page2():

    st.subheader('''Here is the heatmap of all the columns based on the correlation with CO2_Emissions''')

    cdf = df[['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Inflight service','Cleanliness','Departure Delay in Minutes','Arrival Delay in Minutes']]

    labels = df.columns

    if "hue" not in st.session_state:
        st.session_state = "satisfaction"


    st.radio(
        "Set hue: ",
        key = "hue",
        options = ["satisfaction"]
    )

    sns.set(rc = {'figure.figsize':(20,20)})
    sns.set(font_scale=2.5)
    sns.heatmap(cdf.corr(), annot = True, annot_kws={"size":20})
    st.pyplot(plt)

#     def aggrid_interactive_table(df: pd.DataFrame):
#         options = GridOptionsBuilder.from_dataframe(
#             df, enableRowGroup=True, enableValue=True, enablePivot=True
#         )

#         options.configure_side_bar()

#         options.configure_selection("single")
#         selection = AgGrid(
#             df,
#             enable_enterprise_modules=True,
#             gridOptions=options.build(),
#             height = 500,
#             update_mode=GridUpdateMode.MODEL_CHANGED,
#             allow_unsafe_jscode=True,
#         )

#         return selection


#     selection = aggrid_interactive_table(df=df)

#     if selection:
#         st.write("You selected:")
#         st.json(selection["selected_rows"])

    st.subheader('''An interactive plot between columns of the car crash dataset''')

    x_axis_choice = st.selectbox(
        "x axis",
        labels)
    y_axis_choice = st.selectbox(
        "y axis",
        labels)

    crash = alt.Chart(df).mark_circle().encode(
        y = y_axis_choice,
        x = x_axis_choice,
        color='On-board service',
        size='Cleanliness',
        tooltip=['Gender','Age','Customer Type','Type of Travel','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes'],
    ).properties(
        width=400,
        height=300
        ).interactive()

    st.altair_chart(crash)

    plt.figure(figsize= (6,5))
    img_2 = px.histogram(df, x= y_axis_choice, marginal="box")
    st.plotly_chart(img_2)

    plt.figure(figsize=(6,5))
    img_1 = px.histogram(df, x = x_axis_choice)
    st.plotly_chart(img_1)

    st.bar_chart(df,x=x_axis_choice,y=y_axis_choice)

def page3():

    tab1, tab2 = st.tabs(["Classifiers", "Custom Input"])

    with tab1:

        st.header("classification model")
        
        classifier_name = st.sidebar.selectbox('Select classifier',("KNN","SVM","Decision Tree", "Random Forest","Neural Net", "Logistic Regression"))
        
        def add_parameter_ui(clf_name):
            params = dict()
            if clf_name == 'SVM':
                C = st.sidebar.slider('C', 0.01, 10.0, 2.5)
                params['C'] = C
            elif clf_name == 'KNN':
                K = st.sidebar.slider('K', 1, 15, 6)
                params['K'] = K
            elif clf_name == "Decision Tree":
                max_depth = st.sidebar.slider('max_depth', 2, 15, 6)
                params['max_depth'] = max_depth
            elif clf_name == "Neural Net":
                alpha = st.sidebar.slider('alpha', 0.1, 1.0, 0.95)
                params['alpha'] =alpha
                max_iter = st.sidebar.slider('max_iter', 1, 100, 32)
                params['max_iter'] = max_iter
            elif clf_name == "Logistic Regression":
                n_jobs = st.sidebar.slider('alpha', -1, 1, 1)
                params['n_jobs'] =n_jobs
            else:
                max_depth = st.sidebar.slider('max_depth', 2, 15, 7)
                params['max_depth'] = max_depth
                n_estimators = st.sidebar.slider('n_estimators', 1, 100, 25)
                params['n_estimators'] = n_estimators
            return params

        params = add_parameter_ui(classifier_name)

        def get_classifier(clf_name, params):
            clf = None
            if clf_name == 'SVM':
                clf = SVC(C=params['C'])
            elif clf_name == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=params['K'])
            elif clf_name == "Decision Tree":
                clf = DecisionTreeClassifier(max_depth=params['max_depth'])
            elif clf_name == "Neural Net":
                clf = MLPClassifier(alpha= params['alpha'], max_iter= params['max_iter'])
            elif clf_name == "Logistic Regression":
                clf = LogisticRegression(n_jobs = params['n_jobs'])
            else:
                clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)
            return clf

        clf = get_classifier(classifier_name, params)

        X = df.drop('satisfaction', axis =1)
        y = df['satisfaction']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.write(f'Classifier = {classifier_name}')
        st.write(f'Accuracy =', acc)

        def plot_metrics(metrics_list):
            if "Confusion Matrix" in metrics_list:
                st.subheader("Confusion Matrix")
                #plot_confusion_matrix(clf, X_test, y_test, display_labels=   class_names)
                ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels = class_names).plot(cmap = 'gist_heat_r')
                st.pyplot()
            # if "ROC Curve" in metrics_list:
            #     st.subheader("ROC Curve")
            #     plot_roc_curve(clf, X_test, y_test)
            #     st.pyplot()
            # if "Precision-Recall Curve" in metrics_list:
            #     st.subheader("Precision-Recall Curve")
            #     plot_precision_recall_curve(clf, X_test, y_test)
            #     st.pyplot()
        class_names = ["satisfaction", "neutral or dissatisfied"]
        st.set_option('deprecation.showPyplotGlobalUse', False)

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        plot_metrics(metrics)

    with tab2:

        st.header("Custom input")
        Gender= st.radio("Select Gender : 0 is Male, 1 is Female",
        (0,1))


        Customer_Type= st.radio("Loyality : 0 is Loyal Customer, 1 is disloyal Customer",
        (0,1))

        Age= st.slider('Age',10,100)

        Type_of_Travel = st.radio("Type of travel : 0 is Personal Travel, 1 is Business travel",
        (0,1))

        Class= st.radio("Choose class : 0 is Economy Plus, 1 is Business, 2 is Economy",
        (0,1,2))
        
        Flight_Distance = st.slider('Flight Distance',30,5000)

        Departure_Delay = st.slider('Departure Delay in Minutes',0,60)

        Arrival_Delay = st.slider('Arrival Delay in Minutes',0,60)

        st.write("choose how satisfied your are with the below listed services")
        st.write("0 : Not satisfied  -  5 : Fully satisfied ")

        Inflight_wifi_service = st.slider('Inflight wifi service', 0,5)
    
        DepartureArrival_time_convenient = st.slider('Departure/Arrival_time_convenient',0,5)

        Online_booking = st.slider('Ease of Online booking',0,5)

        Gate_location = st.slider('Gate location',0,5)

        Food_drink = st.slider('Food_drink',0,5)

        Online_boarding = st.slider('Online boarding',0,5)

        Seat_comfort = st.slider('Seat comfort',0,5)

        Inflight_entertainment = st.slider('Inflight entertainment',0,5)

        Onboard_service = st.slider('On-board service',0,5)

        Leg_room = st.slider('Leg room service',0,5)

        Baggage_handling = st.slider('Baggage handling',0,5)

        Checkin_service = st.slider('Checkin service',0,5)

        Inflight_service = st.slider('Inflight service',0,5)

        Cleanliness = st.slider('Cleanliness',0,5)

        inputs={
            'Gender':Gender,
            'Customer Type':Customer_Type,
            'Age':Age,
            'Type of Travel':Type_of_Travel,
            'Class':Class,
            'Flight Distance':Flight_Distance,
            'Inflight wifi service':Inflight_wifi_service,
            'Departure/Arrival_time_convenient':DepartureArrival_time_convenient,
            'Online booking':Online_booking,
            'Gate_location':Gate_location,
            'Food_drink':Food_drink,
            'Online_boarding':Online_boarding,
            'Seat comfort':Seat_comfort,
            'Inflight entertainment':Inflight_entertainment,
            'Onboard service':Onboard_service,
            'Leg room':Leg_room,
            'Baggage handling':Baggage_handling,
            'Checkin service':Checkin_service,
            'Inflight service':Inflight_service,
            'Cleanliness':Cleanliness,
            'Departure Delay':Departure_Delay,
            'Arrival Delay':Arrival_Delay
            }
        
        fly = pd.DataFrame(inputs, index =[0])

        y_pred = clf.predict(fly)
        st.subheader("Result")
        st.write(y_pred)
        if y_pred==0:
            st.markdown('''### Not Satisfied😗''')
        else:
            st.markdown('''### Satisfied😊''')

page_names_to_funcs = {
    "Introduction": page1,
    "Exploratory Data Analysis": page2,
    "Classifiers": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
