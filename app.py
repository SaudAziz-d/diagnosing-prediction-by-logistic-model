import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# Streamlit app for breast cancer diagnosis prediction
st.set_page_config(page_title="Diagnosis Prediction", layout="wide")


@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('breast_cancer_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'breast_cancer_model.pkl' is in the same directory.")
        return None

@st.cache_resource
def load_scaler():
    """Load or create the scaler for data preprocessing"""
    try:
        scaler = joblib.load('scaler.pkl')
        return scaler
    except FileNotFoundError:
        # If scaler not found, create a new one (you may need to retrain this)
        return StandardScaler()

def preprocess_data(data):
    """Preprocess the uploaded data"""
    # Drop unnecessary columns if they exist
    columns_to_drop = ['Unnamed: 32', 'Unnamed: 33', 'id']
    for col in columns_to_drop:
        if col in data.columns:
            data = data.drop(col, axis=1)
    
    # If diagnosis column exists, convert it to binary
    if 'diagnosis' in data.columns:
        data['diagnosis'] = [1 if value == 'M' else 0 for value in data['diagnosis']]
        y = data['diagnosis']
        X = data.drop(['diagnosis'], axis=1)
        return X, y
    else:
        return data, None

def create_prediction_distribution_chart(predictions):
    """Create a distribution chart of predictions"""
    pred_counts = pd.Series(predictions).value_counts()
    
    fig = px.bar(
        x=['Benign (0)', 'Malignant (1)'],
        y=[pred_counts.get(0, 0), pred_counts.get(1, 0)],
        title="Distribution of Predictions",
        labels={'x': 'Diagnosis', 'y': 'Count'},
        color=['Benign (0)', 'Malignant (1)'],
        color_discrete_map={'Benign (0)': 'lightblue', 'Malignant (1)': 'salmon'}
    )
    return fig

def create_feature_correlation_heatmap(data):
    """Create a correlation heatmap of features"""
    correlation_matrix = data.corr()
    
    fig = px.imshow(
        correlation_matrix.values,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r"
    )
    fig.update_layout(width=800, height=800)
    return fig

def create_confidence_distribution(probabilities):
    """Create a distribution chart of prediction confidence"""
    confidence_scores = np.max(probabilities, axis=1)
    
    fig = px.histogram(
        x=confidence_scores,
        title="Distribution of Prediction Confidence",
        labels={'x': 'Confidence Score', 'y': 'Frequency'},
        nbins=20
    )
    fig.update_layout(xaxis_range=[0.5, 1.0])
    return fig

def generate_prediction_report(data, predictions, probabilities):
    """Generate a comprehensive prediction report"""
    report_data = data.copy()
    report_data['Prediction'] = ['Malignant' if pred == 1 else 'Benign' for pred in predictions]
    report_data['Confidence'] = np.max(probabilities, axis=1)
    report_data['Malignant_Probability'] = probabilities[:, 1]
    report_data['Benign_Probability'] = probabilities[:, 0]
    
    return report_data

def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def main():
    # Load the model and scaler
    model = load_model()
    scaler = load_scaler()
    
    # App title
    st.title("Breast Cancer Diagnosis Prediction")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # File upload option in sidebar
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file containing breast cancer diagnostic data"
    )
    
    # Initialize variables for metrics
    predictions = None
    probabilities = None
    
    # Process uploaded file first to get predictions for summary
    if uploaded_file is not None:
        try:
            # Try different approaches to read the CSV file
            try:
                # First attempt - standard reading
                data = pd.read_csv(uploaded_file)
            except pd.errors.EmptyDataError:
                st.error("The uploaded file appears to be empty.")
                return
            except UnicodeDecodeError:
                # Try different encoding
                uploaded_file.seek(0)  # Reset file pointer
                data = pd.read_csv(uploaded_file, encoding='latin-1')
            except pd.errors.ParserError:
                # Try with different separator
                uploaded_file.seek(0)  # Reset file pointer
                data = pd.read_csv(uploaded_file, sep=';')
            
            # Check if dataframe is empty or has no columns
            if data.empty or len(data.columns) == 0:
                st.error("The uploaded file contains no data or columns.")
                return
                
            # Preprocess the data
            X, y = preprocess_data(data.copy())
            
            # Check if we have features to work with
            if X.empty or len(X.columns) == 0:
                st.error("No valid features found in the uploaded file.")
                return
                
            # Scale the features
            X_scaled = scaler.fit_transform(X)
            # Make predictions
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            predictions = None
            probabilities = None
    
    # Display prediction summary at the top if we have predictions
    if predictions is not None:
        st.subheader("Prediction Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            benign_count = np.sum(predictions == 0)
            st.metric("Benign Cases", benign_count)
        
        with col2:
            malignant_count = np.sum(predictions == 1)
            st.metric("Malignant Cases", malignant_count)
        
        with col3:
            avg_confidence = np.mean(np.max(probabilities, axis=1))
            st.metric("Average Confidence", f"{avg_confidence:.2%}")
        
        st.markdown("---")
    
    # Main content area
    if model is None:
        st.error("Model could not be loaded")
        return
    
    if uploaded_file is not None:
        # Load and display the uploaded data
        try:
            # Try different approaches to read the CSV file
            try:
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                data = pd.read_csv(uploaded_file)
            except pd.errors.EmptyDataError:
                st.error("The uploaded file appears to be empty.")
                return
            except UnicodeDecodeError:
                # Try different encoding
                uploaded_file.seek(0)  # Reset file pointer
                data = pd.read_csv(uploaded_file, encoding='latin-1')
            except pd.errors.ParserError:
                # Try with different separator
                uploaded_file.seek(0)  # Reset file pointer
                data = pd.read_csv(uploaded_file, sep=';')
            
            # Check if dataframe is empty or has no columns
            if data.empty or len(data.columns) == 0:
                st.error("The uploaded file contains no data or columns.")
                return
                
            st.subheader("Uploaded Data Preview")
            st.write(f"Dataset shape: {data.shape}")
            
            # Use expander for data preview
            with st.expander("View Uploaded Data Sample", expanded=False):
                st.dataframe(data.head())
            
            # Data Visualization Dashboard
            st.subheader("Data Visualization Dashboard")
            
            # Prediction distribution chart
            st.plotly_chart(create_prediction_distribution_chart(predictions), use_container_width=True)
            
            # Feature correlation heatmap
            if X.shape[1] <= 30:  # Only show if not too many features
                st.plotly_chart(create_feature_correlation_heatmap(X), use_container_width=True)
            
            # Prediction confidence distribution
            st.plotly_chart(create_confidence_distribution(probabilities), use_container_width=True)
            
            # Results Export Section
            st.subheader("Results Export")
            
            # Generate prediction report
            report_data = generate_prediction_report(X, predictions, probabilities)
            
            # Use expander for prediction results
            with st.expander("View Sample Prediction Results", expanded=False):
                st.dataframe(report_data[['Prediction', 'Confidence', 'Malignant_Probability', 'Benign_Probability']].head())
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Download full report as CSV
                csv_data = convert_df_to_csv(report_data)
                st.download_button(
                    label="Download Full Report (CSV)",
                    data=csv_data,
                    file_name=f"breast_cancer_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download predictions only
                predictions_only = pd.DataFrame({
                    'Prediction': ['Malignant' if pred == 1 else 'Benign' for pred in predictions],
                    'Confidence': np.max(probabilities, axis=1)
                })
                predictions_csv = convert_df_to_csv(predictions_only)
                st.download_button(
                    label="Download Predictions Only (CSV)",
                    data=predictions_csv,
                    file_name=f"predictions_only_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted and contains the required columns.")
    
    else:
        st.info("Please upload a CSV file to get started with predictions and visualizations.")
        
        # Show sample data format
        st.subheader("Expected Data Format")
        st.write("Your CSV file should contain the following types of columns:")
        sample_cols = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', '...']
        st.code(", ".join(sample_cols))
        st.write("The file should contain the same features used to train the model.")

if __name__ == "__main__":
    main() 