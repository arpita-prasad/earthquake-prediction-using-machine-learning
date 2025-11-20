import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Earthquake Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .stApp {
        background: transparent;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .severity-low {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    }
    .severity-moderate {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    .severity-high {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    }
    .severity-extreme {
        background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Header and description
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: black;'>🌍 Earthquake Prediction System</h1>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar
st.sidebar.markdown("## 📊 Input Parameters")
st.sidebar.markdown("Configure the earthquake parameters below:")

with st.sidebar.expander("🌐 Location Parameters", expanded=True):
    latitude = st.number_input("Latitude (°)", min_value=-90.0, max_value=90.0, value=35.0, step=0.01, help="Geographic latitude coordinate")
    longitude = st.number_input("Longitude (°)", min_value=-180.0, max_value=180.0, value=80.0, step=0.01, help="Geographic longitude coordinate")
    depth = st.number_input("Depth (km)", min_value=0.0, max_value=500.0, value=10.0, step=0.1, help="Depth below Earth's surface")

with st.sidebar.expander("📅 Temporal Parameters", expanded=True):
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2025, step=1)
    month = st.selectbox("Month", list(range(1, 13)), index=0)
    day = st.selectbox("Day", list(range(1, 32)), index=0)
    hour = st.selectbox("Hour (UTC)", list(range(0, 24)), index=0)
    weekday = st.selectbox(
        "Weekday",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        index=0
    )

# Convert weekday to numeric
weekday_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
weekday_num = weekday_map[weekday]

# Preparing input DataFrame
feature_order = ["latitude", "longitude", "depth", "hour", "year", "day", "month", "weekday"]
input_data = {
    "latitude": latitude, "longitude": longitude, "depth": depth, "hour": hour,
    "year": year, "day": day, "month": month, "weekday": weekday_num
}
input_df = pd.DataFrame([input_data])[feature_order]

# Display input summary
st.markdown("### 📍 Current Input Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Latitude", f"{latitude}°")
with col2:
    st.metric("Longitude", f"{longitude}°")
with col3:
    st.metric("Depth", f"{depth} km")
with col4:
    st.metric("Date", f"{year}-{month:02d}-{day:02d}")

# Prediction button with enhanced styling
predict_button = st.sidebar.button("Predict Magnitude", type="primary", use_container_width=True)

# Load model and predict
model_path = "lightgbm_earthquake_model.pkl"
if not os.path.exists(model_path):
    st.error("⚠️ Model file not found. Please ensure 'lightgbm_earthquake_model.pkl' is present.")
else:
    try:
        model = joblib.load(model_path)

        if predict_button:
            with st.spinner("🔄 Analyzing seismic parameters..."):
                try:
                    prediction = model.predict(input_df)[0]
                    
                    # Determine severity class
                    if prediction < 4.0:
                        severity_class = "severity-low"
                        severity_text = "Minor"
                        severity_emoji = "🟢"
                    elif prediction < 5.5:
                        severity_class = "severity-moderate"
                        severity_text = "Moderate"
                        severity_emoji = "🟡"
                    elif prediction < 7.0:
                        severity_class = "severity-high"
                        severity_text = "Major"
                        severity_emoji = "🟠"
                    else:
                        severity_class = "severity-extreme"
                        severity_text = "Extreme"
                        severity_emoji = "🔴"
                    
                    # Display prediction with custom styling
                    st.markdown(f"""
                        <div class='prediction-box {severity_class}'>
                            {severity_emoji} Predicted Magnitude: {prediction:.2f}<br>
                            <span style='font-size: 20px;'>Severity: {severity_text}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Three column layout for visualizations
                    st.markdown("---")
                    st.markdown("## 📊 Detailed Analysis")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Feature Importance", "📈 Depth Analysis", "🌏 Geographic View", "📚 Educational Insights"])
                    
                    with tab1:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            # Feature Importance
                            importances = model.feature_importances_
                            features = input_df.columns
                            fi_df = pd.DataFrame({
                                "Feature": features,
                                "Importance": importances
                            }).sort_values(by="Importance", ascending=True)
                            
                            fig = px.bar(
                                fi_df, 
                                x="Importance", 
                                y="Feature", 
                                orientation="h",
                                title="Feature Importance in Prediction Model",
                                color="Importance",
                                color_continuous_scale="Viridis"
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("### 🔍 Key Insights")
                            st.markdown(f"""
                            - **Most Important**: {fi_df.iloc[-1]['Feature'].title()}
                            - **Importance Score**: {fi_df.iloc[-1]['Importance']:.3f}
                            - **Least Important**: {fi_df.iloc[0]['Feature'].title()}
                            
                            Feature importance indicates which parameters have the strongest influence on earthquake magnitude predictions.
                            """)
                    
                    with tab2:
                        # Depth vs Magnitude comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Create depth distribution comparison
                            depth_range = np.linspace(0, 500, 50)
                            magnitude_predictions = []
                            
                            for d in depth_range:
                                temp_input = input_df.copy()
                                temp_input['depth'] = d
                                pred = model.predict(temp_input)[0]
                                magnitude_predictions.append(pred)
                            
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(
                                x=depth_range,
                                y=magnitude_predictions,
                                mode='lines',
                                name='Predicted Magnitude',
                                line=dict(color='cyan', width=3)
                            ))
                            fig2.add_trace(go.Scatter(
                                x=[depth],
                                y=[prediction],
                                mode='markers',
                                name='Your Input',
                                marker=dict(size=15, color='red', symbol='star')
                            ))
                            fig2.update_layout(
                                title="Magnitude vs Depth Relationship",
                                xaxis_title="Depth (km)",
                                yaxis_title="Predicted Magnitude",
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        with col2:
                            # Gauge chart for magnitude
                            fig3 = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=prediction,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Magnitude Scale"},
                                delta={'reference': 5.0},
                                gauge={
                                    'axis': {'range': [None, 10]},
                                    'bar': {'color': "darkred"},
                                    'steps': [
                                        {'range': [0, 4], 'color': "lightgreen"},
                                        {'range': [4, 5.5], 'color': "yellow"},
                                        {'range': [5.5, 7], 'color': "orange"},
                                        {'range': [7, 10], 'color': "red"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 7
                                    }
                                }
                            ))
                            fig3.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white', size=16)
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                    
                    with tab3:
                        # Geographic visualization
                        fig4 = go.Figure(go.Scattergeo(
                            lon=[longitude],
                            lat=[latitude],
                            text=f"Magnitude: {prediction:.2f}",
                            mode='markers+text',
                            marker=dict(
                                size=prediction*5,
                                color=prediction,
                                colorscale='Reds',
                                showscale=True,
                                colorbar=dict(title="Magnitude")
                            )
                        ))
                        fig4.update_layout(
                            title="Earthquake Location",
                            geo=dict(
                                projection_type='natural earth',
                                showland=True,
                                landcolor='rgb(243, 243, 243)',
                                coastlinecolor='rgb(204, 204, 204)',
                            ),
                            height=500
                        )
                        st.plotly_chart(fig4, use_container_width=True)
                        
                        # Location info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Coordinates", f"{latitude}°, {longitude}°")
                        with col2:
                            st.metric("Depth Classification", 
                                     "Shallow" if depth < 70 else "Intermediate" if depth < 300 else "Deep")
                        with col3:
                            st.metric("Time", f"{hour:02d}:00 UTC")
                    
                    with tab4:
                        st.markdown("### 🌋 Earthquake Magnitude Scale (Richter Scale)")
                        
                        magnitude_info = pd.DataFrame({
                            "Magnitude": ["< 2.0", "2.0 - 3.9", "4.0 - 4.9", "5.0 - 5.9", "6.0 - 6.9", "7.0 - 7.9", "8.0+"],
                            "Classification": ["Micro", "Minor", "Light", "Moderate", "Strong", "Major", "Great"],
                            "Effects": [
                                "Not felt, recorded by seismographs",
                                "Often felt, rarely causes damage",
                                "Noticeable shaking, minimal damage",
                                "Can damage poorly constructed buildings",
                                "Destructive in populated areas",
                                "Serious damage over large areas",
                                "Devastating damage over vast areas"
                            ]
                        })
                        st.dataframe(magnitude_info, use_container_width=True)
                        
                        st.markdown("### 📖 Key Earthquake Facts")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            #### Energy Release
                            - Each whole number increase on the Richter scale represents roughly **32 times** more energy released
                            - A magnitude 6.0 earthquake releases energy equivalent to about **15 kilotons of TNT**
                            
                            #### Depth Classification
                            - **Shallow**: 0-70 km (most destructive)
                            - **Intermediate**: 70-300 km
                            - **Deep**: 300-700 km (less surface impact)
                            
                            #### Global Statistics
                            - Approximately **500,000** detectable earthquakes occur annually
                            - Only about **100** cause damage each year
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            #### Your Prediction Analysis
                            - **Predicted Magnitude**: {prediction:.2f}
                            - **Severity Level**: {severity_text}
                            - **Depth Category**: {'Shallow' if depth < 70 else 'Intermediate' if depth < 300 else 'Deep'}
                            - **Energy Equivalent**: ~{32**(prediction-4):.1f}x a magnitude 4.0 quake
                            
                            #### Safety Recommendations
                            - {f"🟢 Minor earthquake - minimal safety concerns" if prediction < 4 else ""}
                            - {f"🟡 Moderate earthquake - stay alert and prepared" if 4 <= prediction < 5.5 else ""}
                            - {f"🟠 Major earthquake - ensure emergency preparedness" if 5.5 <= prediction < 7 else ""}
                            - {f"🔴 Extreme earthquake - critical safety protocols required" if prediction >= 7 else ""}
                            """)
                        
                        st.markdown("### 🛡️ Earthquake Preparedness Tips")
                        st.markdown("""
                        1. **Before**: Secure heavy items, create emergency kit, plan evacuation routes
                        2. **During**: Drop, Cover, and Hold On - get under sturdy furniture
                        3. **After**: Check for injuries, inspect damage, be prepared for aftershocks
                        4. **Emergency Kit**: Water, food, flashlight, first aid, radio, batteries
                        """)

                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
        else:
            st.info("Configure the parameters in the sidebar and click **Predict Magnitude** to get started!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class='info-box'>
                    <h3>🎯 Data-driven Predictions</h3>
                    <p>Powered by a LightGBM model trained on historical seismic patterns.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='info-box'>
                    <h3>📊 Visual Analytics</h3>
                    <p>Comprehensive visualizations and insights to understand seismic patterns.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class='info-box'>
                    <h3>🌍 Global Coordinates</h3>
                    <p>The model accepts locations worldwide, but is optimized for the Indian region.</p>
                </div>
                """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")