import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Advertising Sales Prediction",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>

/* ==========================================
            GOOGLE FONT
========================================== */

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
}

/* ==========================================
            MAIN BACKGROUND
========================================== */

.stApp{
    background: linear-gradient(135deg,#F8FBFF,#EEF6FF,#F9FCFF);
}


/* ==========================================
            HEADER
========================================== */

.dashboard-header{

    background: linear-gradient(135deg,#4F8EF7,#6CC3FF,#86E3FF);

    padding:35px;

    border-radius:25px;

    text-align:center;

    color:white;

    margin-bottom:35px;

    box-shadow:0px 12px 30px rgba(0,102,255,.25);

    animation:fadeDown 1s;
}

.dashboard-header h1{

    font-size:42px;

    font-weight:700;

    margin-bottom:10px;

}

.dashboard-header p{

    font-size:18px;

    opacity:.95;

}

.dashboard-header:hover{

    transform:translateY(-4px);

    transition:.4s;

}


/* ==========================================
            SIDEBAR
========================================== */

section[data-testid="stSidebar"]{

    background:linear-gradient(180deg,#EAF5FF,#D6EEFF);

    border-right:2px solid #B7DBFF;

}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] label{

    color:#1565C0;

    font-weight:600;

}


/* ==========================================
            KPI METRIC
========================================== */

div[data-testid="metric-container"]{

    background:white;

    border-radius:20px;

    padding:20px;

    border:1px solid #E6EEF7;

    box-shadow:0px 8px 20px rgba(0,0,0,.08);

    transition:.4s;

}

div[data-testid="metric-container"]:hover{

    transform:translateY(-6px);

    box-shadow:0px 18px 35px rgba(33,150,243,.20);

}


/* ==========================================
            BUTTON
========================================== */

.stButton>button{

    width:100%;

    background:linear-gradient(90deg,#2196F3,#42A5F5);

    color:white;

    border:none;

    border-radius:12px;

    padding:12px;

    font-size:17px;

    font-weight:600;

    transition:.4s;

}

.stButton>button:hover{

    background:linear-gradient(90deg,#1976D2,#1565C0);

    transform:scale(1.03);

}


/* ==========================================
            INPUTS
========================================== */

.stTextInput input,
.stNumberInput input{

    border-radius:12px;

    border:1px solid #C5DDF6;

}

.stSelectbox div{

    border-radius:12px;

}


/* ==========================================
            SLIDER
========================================== */

.stSlider{

    padding-top:12px;

}


/* ==========================================
            DATAFRAME
========================================== */

div[data-testid="stDataFrame"]{

    background:white;

    border-radius:18px;

    padding:15px;

    box-shadow:0px 8px 18px rgba(0,0,0,.08);

}


/* ==========================================
            PLOTLY CHARTS
========================================== */

.js-plotly-plot{

    background:white;

    border-radius:20px;

    padding:15px;

    box-shadow:0px 8px 20px rgba(0,0,0,.08);

}


/* ==========================================
            PAGE TITLE
========================================== */

.page-title{

    font-size:34px;

    font-weight:700;

    color:#1976D2;

    margin-bottom:20px;

}


/* ==========================================
            SECTION CARD
========================================== */

.card{

    background:white;

    padding:25px;

    border-radius:20px;

    box-shadow:0px 8px 20px rgba(0,0,0,.08);

    margin-bottom:20px;

}


/* ==========================================
            PREDICTION RESULT
========================================== */

.prediction-box{

    background:linear-gradient(135deg,#E3FCEC,#C8F7DC);

    padding:35px;

    border-radius:25px;

    text-align:center;

    box-shadow:0px 10px 25px rgba(76,175,80,.18);

}

.prediction-box h2{

    color:#2E7D32;

}

.prediction-box h1{

    font-size:55px;

    color:#1B5E20;

}


/* ==========================================
            MODEL RESULT
========================================== */

.model-box{

    background:white;

    border-left:6px solid #2196F3;

    padding:20px;

    border-radius:18px;

    box-shadow:0px 6px 15px rgba(0,0,0,.08);

}


/* ==========================================
            EXPANDER
========================================== */

.streamlit-expanderHeader{

    font-size:18px;

    font-weight:600;

    color:#1565C0;

}


/* ==========================================
            SCROLLBAR
========================================== */

::-webkit-scrollbar{

    width:10px;

}

::-webkit-scrollbar-thumb{

    background:#90CAF9;

    border-radius:20px;

}

::-webkit-scrollbar-track{

    background:#F5F5F5;

}


/* ==========================================
            ANIMATION
========================================== */

@keyframes fadeDown{

    from{

        opacity:0;

        transform:translateY(-40px);

    }

    to{

        opacity:1;

        transform:translateY(0);

    }

}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown('<p class="title">📊 Advertising Sales Prediction Dashboard</p>',unsafe_allow_html=True)

st.markdown('<p class="subtitle">Interactive Machine Learning Dashboard using Streamlit</p>',unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df=pd.read_csv("advertising.csv")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Navigation")

option=st.sidebar.radio(
    "Choose Section",
    ["Dataset","Visualization","Model","Prediction"]
)

# --------------------------------------------------
# Dataset
# --------------------------------------------------

if option=="Dataset":

    st.header("Dataset Preview")

    st.dataframe(df,use_container_width=True)

    st.subheader("Dataset Statistics")

    st.dataframe(df.describe())

# --------------------------------------------------
# Visualization
# --------------------------------------------------

elif option == "Visualization":

    st.markdown("""
    <div class="card">
        <h2 style='color:#1565C0;'>📊 Interactive Visualization Dashboard</h2>
        <p style='color:gray;'>
            Analyze the relationship between advertising budget and sales using
            interactive charts.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    feature = st.selectbox(
        "📌 Select Advertisement Feature",
        ["TV", "Radio", "Newspaper"]
    )

    c1, c2 = st.columns([2, 1])

    with c1:

        fig = px.scatter(
            df,
            x=feature,
            y="Sales",
            color="Sales",
            size="Sales",
            
            template="plotly_white",
            title=f"{feature} Advertisement vs Sales"
        )

        fig.update_layout(
            height=450,
            title_x=0.35,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=15)
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:

        st.markdown("### 📈 Statistics")

        st.metric("Maximum", round(df[feature].max(),2))

        st.metric("Average", round(df[feature].mean(),2))

        st.metric("Minimum", round(df[feature].min(),2))

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        fig = px.histogram(
            df,
            x=feature,
            nbins=25,
            color_discrete_sequence=["#42A5F5"],
            template="plotly_white",
            title=f"{feature} Distribution"
        )

        fig.update_layout(height=400)

        st.plotly_chart(fig,use_container_width=True)

    with right:

        fig = px.box(
            df,
            y=feature,
            color_discrete_sequence=["#26A69A"],
            template="plotly_white",
            title=f"{feature} Box Plot"
        )

        fig.update_layout(height=400)

        st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")

    st.subheader("🔥 Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    heat = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        template="plotly_white"
    )

    heat.update_layout(height=500)

    st.plotly_chart(heat,use_container_width=True)

    st.markdown("---")

    col1,col2=st.columns(2)

    with col1:

        fig = px.scatter_matrix(
            df,
            dimensions=["TV","Radio","Newspaper","Sales"],
            color="Sales",
            title="Relationship Between Features"
        )

        fig.update_layout(height=650)

        st.plotly_chart(fig,use_container_width=True)

    with col2:

        fig = px.pie(
            names=["TV","Radio","Newspaper"],
            values=[
                df["TV"].sum(),
                df["Radio"].sum(),
                df["Newspaper"].sum()
            ],
            title="Advertisement Budget Distribution",
            hole=0.5,
            color_discrete_sequence=[
                "#42A5F5",
                "#26C6DA",
                "#66BB6A"
            ]
        )

        st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")

    fig = px.line(
        df,
        y="Sales",
        title="Sales Trend",
        markers=True,
        template="plotly_white"
    )

    fig.update_layout(
        height=450,
        title_x=0.45
    )

    st.plotly_chart(fig,use_container_width=True)

# --------------------------------------------------
# Model
# --------------------------------------------------

elif option == "Model":

    st.markdown("""
    <div class="card">
        <h2 style='color:#1565C0;'>🤖 Machine Learning Model Performance</h2>
        <p style='color:gray;'>
            Evaluate the Linear Regression model using different performance metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Prepare Data
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mse = mean_squared_error(y_test, prediction)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, prediction)

    accuracy = r2 * 100

    # KPI Cards

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🎯 Accuracy", f"{accuracy:.2f}%")

    with c2:
        st.metric("📉 RMSE", f"{rmse:.2f}")

    with c3:
        st.metric("📊 MSE", f"{mse:.2f}")

    with c4:
        st.metric("⭐ R² Score", f"{r2:.3f}")

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        coef = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_
        })

        fig = px.bar(
            coef,
            x="Feature",
            y="Coefficient",
            color="Coefficient",
            text="Coefficient",
            template="plotly_white",
            title="Feature Importance"
        )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

    with right:

        fig = px.scatter(
            x=y_test,
            y=prediction,
            
            labels={
                "x": "Actual Sales",
                "y": "Predicted Sales"
            },
            template="plotly_white",
            title="Actual vs Predicted"
        )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    residual = y_test - prediction

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            residual,
            nbins=20,
            title="Residual Distribution",
            template="plotly_white",
            color_discrete_sequence=["#42A5F5"]
        )

        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.scatter(
            x=prediction,
            y=residual,
            labels={
                "x": "Predicted Sales",
                "y": "Residual"
            },
            title="Residual Plot",
            template="plotly_white"
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="red"
        )

        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📋 Model Summary")

    st.info(f"""
### Linear Regression Equation

Sales =
({model.coef_[0]:.3f}) × TV +
({model.coef_[1]:.3f}) × Radio +
({model.coef_[2]:.3f}) × Newspaper +
({model.intercept_:.3f})

The model predicts sales using advertising budgets from
TV, Radio and Newspaper.
""")
# --------------------------------------------------
# Prediction
# --------------------------------------------------

else:

    st.markdown("""
    <div class="card">
        <h2 style='color:#1565C0;'>🎯 Sales Prediction Center</h2>
        <p style='color:gray;'>
            Adjust the advertisement budget and predict the expected sales.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Train Model

    X = df[["TV","Radio","Newspaper"]]
    y = df["Sales"]

    model = LinearRegression()
    model.fit(X,y)

    st.markdown("### 💰 Advertisement Budget")

    c1,c2,c3 = st.columns(3)

    with c1:
        tv = st.slider(
            "📺 TV",
            0.0,
            300.0,
            150.0
        )

    with c2:
        radio = st.slider(
            "📻 Radio",
            0.0,
            50.0,
            25.0
        )

    with c3:
        newspaper = st.slider(
            "📰 Newspaper",
            0.0,
            120.0,
            40.0
        )

    st.markdown("---")

    total = tv + radio + newspaper

    k1,k2,k3,k4 = st.columns(4)

    with k1:
        st.metric("📺 TV Budget", f"${tv:.1f}")

    with k2:
        st.metric("📻 Radio Budget", f"${radio:.1f}")

    with k3:
        st.metric("📰 Newspaper Budget", f"${newspaper:.1f}")

    with k4:
        st.metric("💰 Total Budget", f"${total:.1f}")

    st.markdown("---")

    if st.button("🚀 Predict Sales"):

        result = model.predict([[tv,radio,newspaper]])

        prediction = result[0]

        st.markdown(f"""
        <div class="prediction-box">
            <h2>Predicted Sales</h2>
            <h1>{prediction:.2f}</h1>
            <h3>Units</h3>
        </div>
        """, unsafe_allow_html=True)

        left,right = st.columns(2)

        with left:

            gauge = go.Figure(go.Indicator(

                mode="gauge+number",

                value=prediction,

                title={'text':"Sales Prediction"},

                gauge={

                    'axis':{'range':[0,35]},

                    'bar':{'color':"#42A5F5"},

                    'steps':[

                        {'range':[0,10],'color':"#E3F2FD"},

                        {'range':[10,20],'color':"#90CAF9"},

                        {'range':[20,35],'color':"#42A5F5"}

                    ]

                }

            ))

            gauge.update_layout(height=420)

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

        with right:

            pie = px.pie(

                names=["TV","Radio","Newspaper"],

                values=[tv,radio,newspaper],

                hole=.55,

                title="Budget Allocation",

                color_discrete_sequence=[
                    "#42A5F5",
                    "#26C6DA",
                    "#66BB6A"
                ]

            )

            pie.update_layout(height=420)

            st.plotly_chart(
                pie,
                use_container_width=True
            )

        st.markdown("---")

        st.subheader("📈 Budget Contribution")

        contribution = pd.DataFrame({

            "Advertisement":["TV","Radio","Newspaper"],

            "Budget":[tv,radio,newspaper]

        })

        fig = px.bar(

            contribution,

            x="Advertisement",

            y="Budget",

            color="Budget",

            text="Budget",

            template="plotly_white"

        )

        fig.update_layout(height=420)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        if prediction >= 20:
            st.success("✅ Excellent Sales Prediction")

        elif prediction >= 10:
            st.info("📈 Moderate Sales Prediction")

        else:
            st.warning("⚠️ Low Sales Prediction")