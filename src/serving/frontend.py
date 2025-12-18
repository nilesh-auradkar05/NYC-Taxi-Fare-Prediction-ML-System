"""
Streamlit Frontend for NYC Taxi Fare Prediction
===============================================

A user-friendly web interface for predicting NYC Taxi fares.
This frontend communicated with FastAPI backend to generate predictions.

Usage:
    # Start the streamlit app
    streamlit run src/serving/frontend.py

    # or with custom port
    streamlit run src/serving/frontend.py --server.port 8501

Prerequisites:
1. Ensure FastAPI backend is running: uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
2. Model must be downloaded and cached: python or python3 src/serving/download_model.py
"""

import os
from datetime import datetime, timezone, timedelta
from tokenize import triple_quoted

import requests
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="NYC Taxi Fare Prediction",
    page_icon=":taxi:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .title-container {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    
    /* Result card styling */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-card h1 {
        font-size: 3rem;
        margin: 0;
    }
    
    .result-card p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Info box styling */
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions

def check_api_status() -> bool:
    """
    Check if the FastAPI backend is running and responsive.

    Returns:
        bool: True if API is responsive, False otherwise
    """

    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("model_loaded", False)
        return False

    except requests.exceptions.RequestException:
        return False


def get_model_info() -> dict:
    """Fetch model information from the API
    
    Returns:
        dict: Model metadata or empty dict if unavailable
    """
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except requests.exceptions.RequestException:
        return {}

def predict_fare(trip_data: dict) -> dict:
    """
    Send prediction request to the API

    parameters:
        trip_data: Dictionary containing trip details
    
    Returns:
        dict: Prediction result or error message
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=trip_data,
            timeout=10,
        )

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False,
                "error": response.json().get("detail", "Unknown error")
            }

    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}

# Main Application
def main():
    """Main streamlit application"""

    # Header
    st.markdown("""
    <div class="title-container">
        <h1> NYC Taxi Fare Predictor </h1>
        <p style="color: #666; font-size: 1.1rem;">
            Estimate your taxi fare before you ride
        </p>
    </div>
    """, unsafe_allow_html=True)

    # API health check
    api_status = check_api_status()

    if not api_status:
        st.error("""
        **API not available**

        The prediction service is not running. Please ensure:
        1. The FastAPI backend is running: `uvicorn src.serving.api:app --port 8000`
        2. The model has been downloaded and cached: `python or python3 src/serving/download_model.py`
        """)
        st.stop()

    # Show model info in sidebar
    with st.sidebar:
        st.header("Model Information")
        model_info = get_model_info()
        if model_info:
            st.write(f"**Model Name:** {model_info.get("model_name", 'N/A')}")
            st.write(f"**Version:** {model_info.get("model_version", 'N/A')}")
            st.write(f"**Type:** {model_info.get("model_type", 'N/A')}")

        st.markdown("---")
        st.markdown("""
        # How to use
        1. Enter your trip details
        2. Click **Predict Fare**
        3. Get instant fare estimate

        ### About
        This predictor uses an XGBoost Model
        trained on historical millions of NYC taxi trips.
        """)

    # Trip Input Form
    st.subheader("    Trip details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Pickup**")
        pickup_date = st.date_input(
            "Date",
            value=datetime.now().date(),
            key="pickup_date",
        )
        pickup_time = st.time_input(
            "Time",
            value=datetime.now().time(),
            key="pickup_time",
        )

    with col2:
        st.markdown("**Dropoff**")
        default_dropoff = datetime.now() + timedelta(minutes=30)
        dropoff_date = st.date_input(
            "Date",
            value=default_dropoff.date(),
            key="dropoff_date",
        )
        dropoff_time = st.time_input(
            "Time",
            value=default_dropoff.time(),
            key="dropoff_time",
        )

    # Combine date and time
    pickup_datetime = datetime.combine(pickup_date, pickup_time)
    dropoff_datetime = datetime.combine(dropoff_date, dropoff_time)

    # Validate datetime
    if dropoff_datetime <= pickup_datetime:
        st.error("Dropoff time must be after pickup time")

    st.markdown("---")

    # Trip distance
    st.markdown("**Trip Distance**")
    trip_distance = st.slider(
        "Distance (miles)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.1,
        help="Estimated distance of the trip in miles"
    )

    # Calculated trip duration
    trip_duration = (dropoff_datetime - pickup_datetime).total_seconds() / 60
    if trip_duration > 0:
        avg_speed = (trip_distance / trip_duration) * 60
        st.caption(f"Trip Duration: {trip_duration:.0f} min | Avg Speed: {avg_speed:.1f} mph")

    st.markdown("---")

    with st.expander("    Advanced Options"):
        col1, col2 = st.columns(2)

        with col1:
            passenger_count = st.number_input(
                "Passengers",
                min_value=1,
                max_value=4,
                value=1,
                help="Number of passengers (1-4)"
            )

            payment_type = st.selectbox(
                "Payment Type",
                options=[
                    (1, "Credit Card"),
                    (2, "Cash"),
                    (3, "No Charge"),
                    (4, "Dispute")
                ],
                format_func=lambda x: x[1],
                help="How will you pay?"
            )

        with col2:
            rate_code = st.selectbox(
                "Rate Type",
                options=[
                    (1, "Standard"),
                    (2, "JFK"),
                    (3, "Newark"),
                    (4, "Nassau/Westchester"),
                    (5, "Negotiated"),
                    (6, "Group")
                ],
                format_func=lambda x: x[1],
                help="Rate code for the trip"
            )

            vendor = st.selectbox(
                "Vendor",
                options=[
                    (1, "Creative mobile"),
                    (2, "VeriFone"),
                ],
                format_func=lambda x: x[1],
                help="Taxi vendor"
            )

    st.markdown("---")

    # Prediciton Button
    predict_button = st.button(
        "Predict fare",
        type="primary",
        use_container_width=True,
    )

    if predict_button:
        # validate inputs
        if dropoff_datetime <= pickup_datetime:
            st.error("Dropoff time must be after pickup time")
            st.stop()

        # Prepare trip data
        trip_data = {
            "pickup_datetime": pickup_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "dropoff_datetime": dropoff_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "trip_distance": trip_distance,
            "passenger_count": passenger_count,
            "VendorID": vendor[0],
            "RatecodeID": rate_code[0],
            "payment_type": payment_type[0],
            "store_and_fwd_flag": "N",
            "fare_amount": 0.0,
            "tip_amount": 0.0,
            "tolls_amount": 0.0
        }

        # Show loading spinner
        with st.spinner("Calculating fare..."):
            result = predict_fare(trip_data)

        # Display results
        if result["success"]:
            data = result["data"]
            predicted_fare = data["predicted_fare"]

            # Main result card
            st.markdown(f"""
            <div class="result-card">
                <p>Estimated Fare</p>
                <h1>${predicted_fare:.2f}</h1>
                <p>Trip Duration: {data["trip_duration_minutes"]:.0f} minutes</p>
            </div>
            """, unsafe_allow_html=True)

            # Additional details
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Distance", f"{trip_distance:.1f} mi")

            with col2:
                fare_per_mile = predicted_fare / trip_distance if trip_distance > 0 else 0.0
                st.metric("Per mile", f"${fare_per_mile:.2f}")

            with col3:
                st.metric("Model Version", data["model_version"])

            # Tip suggestion
            st.markdown("---")
            st.markdown("**💵 Suggested Tips**")
            tip_col1, tip_col2, tip_col3 = st.columns(3)
            
            with tip_col1:
                tip_15 = predicted_fare * 0.15
                st.info(f"15%: ${tip_15:.2f}\n\nTotal: ${predicted_fare + tip_15:.2f}")
            
            with tip_col2:
                tip_20 = predicted_fare * 0.20
                st.success(f"20%: ${tip_20:.2f}\n\nTotal: ${predicted_fare + tip_20:.2f}")
            
            with tip_col3:
                tip_25 = predicted_fare * 0.25
                st.info(f"25%: ${tip_25:.2f}\n\nTotal: ${predicted_fare + tip_25:.2f}")
                
        else:
            st.error(f"Prediction failed: {result['error']}")
    
    # -------------------------------------------------------------------------
    # FOOTER
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="footer">
        <p>NYC Taxi Fare Prediction Project</p>
    </div>
    """, unsafe_allow_html=True)

# Entry point
if __name__ == "__main__":
    main()