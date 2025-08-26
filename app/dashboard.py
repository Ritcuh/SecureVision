import streamlit as st
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import pandas as pd
from datetime import datetime, timedelta

# Import our modules
from models.utils import load_image_from_upload, load_text_file_from_upload
from app.logic_layer import score_risk, get_owasp_mapping, generate_recommendations
from models.nlp_model import classify_code, analyze_code_vulnerabilities
from models.vision_model import build_vision_model, analyze_screenshot

def create_risk_heatmap(vision_score, nlp_score, combined_risk):
    """Create a risk heatmap visualization."""
    categories = ['Vision Analysis', 'Code Analysis', 'Combined Risk']
    scores = [vision_score, nlp_score, combined_risk]
    
    fig = go.Figure(data=go.Bar(
        x=categories,
        y=scores,
        marker_color=['red' if s > 0.7 else 'orange' if s > 0.4 else 'green' for s in scores],
        text=[f'{s:.2f}' for s in scores],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Security Risk Assessment",
        yaxis_title="Risk Score (0-1)",
        xaxis_title="Analysis Type",
        showlegend=False,
        height=400
    )
    
    return fig

def create_vulnerability_timeline():
    """Create a mock vulnerability evolution timeline."""
    # Generate sample data for demonstration
    dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
    risk_levels = np.random.choice([0.2, 0.5, 0.8], size=30, p=[0.6, 0.3, 0.1])
    
    df = pd.DataFrame({
        'Date': dates,
        'Risk_Level': risk_levels,
        'Vulnerability_Count': np.random.poisson(2, 30)
    })
    
    fig = px.line(df, x='Date', y='Risk_Level', 
                  title='Risk Evolution Over Time',
                  labels={'Risk_Level': 'Risk Score', 'Date': 'Date'})
    
    fig.add_scatter(x=df['Date'], y=df['Risk_Level'], 
                   mode='markers', name='Risk Points',
                   marker=dict(size=df['Vulnerability_Count']*3, 
                             color=df['Risk_Level'], 
                             colorscale='RdYlGn_r'))
    
    return fig

def run_dashboard():
    """Main dashboard function."""
    
    # Sidebar
    st.sidebar.title("🔒 SecureVision")
    st.sidebar.markdown("---")
    
    analysis_type = st.sidebar.selectbox(
        "Analysis Mode",
        ["Full Analysis", "Code Only", "Screenshot Only", "Batch Analysis"]
    )
    
    # Main content
    st.title("SecureVision: AI-Powered Web Security Analysis")
    st.markdown("""
    **Hybrid AI system for detecting web application vulnerabilities through code analysis and UI inspection.**
    
    Upload your web application screenshots and code snippets to get comprehensive security assessments aligned with OWASP standards.
    """)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analysis", "📊 Dashboard", "📋 Reports", "⚙️ Settings"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Screenshot Analysis")
            image_file = st.file_uploader(
                "Upload a web application screenshot", 
                type=["png", "jpg", "jpeg"],
                help="Upload a screenshot of your web application interface"
            )
            
            if image_file:
                st.image(image_file, caption="Uploaded Screenshot", use_column_width=True)
        
        with col2:
            st.subheader("💻 Code Analysis")
            code_input_method = st.radio("Code Input Method", ["Upload File", "Paste Code"])
            
            if code_input_method == "Upload File":
                code_file = st.file_uploader(
                    "Upload code file", 
                    type=["txt", "html", "js", "php", "py", "java"],
                    help="Upload your source code file for analysis"
                )
                code_content = load_text_file_from_upload(code_file) if code_file else None
            else:
                code_content = st.text_area(
                    "Paste your code here:",
                    height=200,
                    placeholder="Enter your HTML, JavaScript, PHP, or other web code..."
                )
        
        # Analysis button
        if st.button("🚀 Run Security Analysis", type="primary", use_container_width=True):
            if (analysis_type in ["Full Analysis"] and image_file and code_content) or \
               (analysis_type == "Code Only" and code_content) or \
               (analysis_type == "Screenshot Only" and image_file):
                
                with st.spinner("Analyzing security vulnerabilities..."):
                    # Initialize models
                    vision_model = build_vision_model()
                    
                    # Perform analyses
                    vision_result = None
                    nlp_result = None
                    
                    if image_file and analysis_type != "Code Only":
                        image_array = load_image_from_upload(image_file)
                        vision_result = analyze_screenshot(vision_model, image_array)
                    
                    if code_content and analysis_type != "Screenshot Only":
                        nlp_result = analyze_code_vulnerabilities(code_content)
                    
                    # Calculate combined risk
                    risk_assessment = score_risk(vision_result, nlp_result)
                    
                    # Display results
                    st.success("Analysis completed!")
                    
                    # Results section
                    st.subheader("🎯 Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if vision_result:
                            st.metric(
                                "Vision Analysis", 
                                f"{vision_result['confidence']:.1%}",
                                f"Risk: {vision_result['risk_level']}"
                            )
                    
                    with col2:
                        if nlp_result:
                            st.metric(
                                "Code Analysis", 
                                f"{nlp_result['confidence']:.1%}",
                                f"Risk: {nlp_result['risk_level']}"
                            )
                    
                    with col3:
                        st.metric(
                            "Overall Risk", 
                            risk_assessment['level'],
                            f"Score: {risk_assessment['score']:.2f}"
                        )
                    
                    # Detailed results
                    if nlp_result and 'vulnerabilities' in nlp_result:
                        st.subheader("🚨 Detected Vulnerabilities")
                        for vuln in nlp_result['vulnerabilities']:
                            with st.expander(f"{vuln['type']} - {vuln['severity']} Risk"):
                                st.write(f"**Description:** {vuln['description']}")
                                st.write(f"**OWASP Category:** {vuln['owasp_category']}")
                                st.code(vuln.get('code_snippet', 'N/A'))
                    
                    # Recommendations
                    recommendations = generate_recommendations(risk_assessment, nlp_result)
                    if recommendations:
                        st.subheader("💡 Security Recommendations")
                        for i, rec in enumerate(recommendations, 1):
                            st.write(f"{i}. **{rec['title']}**")
                            st.write(f"   {rec['description']}")
            else:
                st.error("Please upload the required files based on your selected analysis mode.")
    
    with tab2:
        st.subheader("📊 Security Dashboard")
        
        # Create sample data for dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk heatmap
            if 'risk_assessment' in locals():
                vision_score = vision_result['confidence'] if vision_result else 0.5
                nlp_score = nlp_result['confidence'] if nlp_result else 0.5
                combined_score = risk_assessment['score']
                
                fig_heatmap = create_risk_heatmap(vision_score, nlp_score, combined_score)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Run an analysis to see risk visualization")
        
        with col2:
            # Vulnerability timeline
            fig_timeline = create_vulnerability_timeline()
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # OWASP Top 10 tracking
        st.subheader("🎯 OWASP Top 10 Vulnerability Tracking")
        owasp_data = {
            'Vulnerability': ['Injection', 'Broken Authentication', 'Sensitive Data Exposure', 
                            'XXE', 'Broken Access Control', 'Security Misconfiguration',
                            'XSS', 'Insecure Deserialization', 'Known Vulnerabilities', 'Logging'],
            'Count': [3, 1, 2, 0, 4, 1, 5, 0, 2, 1],
            'Risk Level': ['High', 'Medium', 'High', 'Low', 'Critical', 'Medium',
                          'High', 'Low', 'Medium', 'Low']
        }
        
        df_owasp = pd.DataFrame(owasp_data)
        st.dataframe(df_owasp, use_container_width=True)
    
    with tab3:
        st.subheader("📋 Security Reports")
        st.write("Generate comprehensive security reports for stakeholders.")
        
        if st.button("Generate Report"):
            st.success("Report generated successfully!")
            
            # Mock report data
            report_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_files_analyzed': 15,
                'vulnerabilities_found': 8,
                'critical_issues': 2,
                'high_issues': 3,
                'medium_issues': 2,
                'low_issues': 1
            }
            
            st.json(report_data)
    
    with tab4:
        st.subheader("⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Settings**")
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
            enable_vision = st.checkbox("Enable Vision Analysis", True)
            enable_nlp = st.checkbox("Enable NLP Analysis", True)
        
        with col2:
            st.write("**Alert Settings**")
            email_alerts = st.checkbox("Email Alerts", False)
            slack_integration = st.checkbox("Slack Integration", False)
            auto_reports = st.checkbox("Automatic Reports", False)
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")
    
    # Footer
    st.markdown("---")
    st.markdown("**SecureVision** - AI-Powered Web Security Analysis")
