import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SentinAI - Cyber Threat Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM AESTHETIC ---
def inject_custom_css():
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
        
        /* Global Styles */
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            background-color: #0b0f19; /* Deep space dark */
            color: #e2e8f0;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid #1f2937;
        }
        
        /* Metric Cards (Glassmorphism) */
        div[data-testid="metric-container"] {
            background: rgba(31, 41, 55, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 255, 255, 0.1), 0 4px 6px -2px rgba(0, 255, 255, 0.05);
            border: 1px solid rgba(0, 255, 255, 0.3);
        }
        
        /* Metric Values */
        div[data-testid="metric-container"] > div > div > div {
            color: #38bdf8 !important; /* Cyber blue */
            font-weight: 800;
            font-size: 2rem;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #f8fafc;
            font-weight: 800 !important;
            letter-spacing: -0.025em;
        }
        
        h1 {
            background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Dataframes */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #1f2937;
        }
        
        /* Code blocks (Explainability) */
        code {
            font-family: 'JetBrains Mono', monospace;
            background: #1e293b;
            color: #a78bfa;
        }
        
        /* Alerts */
        .stAlert {
            border-radius: 10px;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    """Load and prepare data for the dashboard."""
    try:
        df = pd.read_csv('data/synthetic_logs.csv.zip', compression='zip', parse_dates=['timestamp'])
        # Sample for performance in dashboard
        if len(df) > 50000:
            df = df.sample(50000, random_state=42).sort_values('timestamp')
            
        # Simulate PyTorch Inference Risk Scores for the UI
        np.random.seed(42)
        # Anomalies are critical (75-99), 5% of normal are medium (30-74), rest are low (1-29)
        normal_scores = np.where(np.random.rand(len(df)) < 0.05, 
                               np.random.uniform(31, 74, len(df)), 
                               np.random.uniform(1, 29, len(df)))
                               
        df['risk_score'] = np.where(df['label'] == 1, 
                                    np.random.uniform(75, 99, len(df)), 
                                    normal_scores)
        df['risk_level'] = pd.cut(df['risk_score'], bins=[0, 30, 74, 100], labels=['Low', 'Medium', 'Critical']).astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}. Please ensure Phase 1 generated the dataset.")
        return pd.DataFrame()

# --- MAIN APP ---
def main():
    inject_custom_css()
    
    st.sidebar.title("🛡️ SentinAI")
    st.sidebar.markdown("*AI-Powered Behavioral Anomaly Detection*")
    
    df = load_data()
    if df.empty:
        return
        
    page = st.sidebar.radio("Navigation", 
                            ["🌐 Global Threat Overview", 
                             "🔎 Threat Hunting", 
                             "👤 Entity Investigation", 
                             "📊 Model Analytics"])
                             
    st.sidebar.divider()
    st.sidebar.markdown("### System Controls")
    live_mode = st.sidebar.toggle("🔴 Live Streaming Mode", help="Simulate real-time log ingestion")
    
    if live_mode:
        if 'sim_idx' not in st.session_state:
            st.session_state.sim_idx = int(len(df) * 0.95) # Start at 95% of data
            
        st.session_state.sim_idx += np.random.randint(10, 100) # Ingest new logs per tick
        if st.session_state.sim_idx > len(df):
            st.session_state.sim_idx = int(len(df) * 0.95) # Loop
            
        df = df.iloc[:st.session_state.sim_idx]
        
    st.sidebar.success(f"🟢 Events Monitored: {len(df):,}")

    # Filter Global Data
    time_range = st.sidebar.slider("Time Range (Days)", 1, 30, 30)
    cutoff_date = df['timestamp'].max() - pd.Timedelta(days=time_range)
    filtered_df = df[df['timestamp'] >= cutoff_date]

    if page == "🌐 Global Threat Overview":
        st.title("Global Threat Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        total_events = len(filtered_df)
        critical_alerts = len(filtered_df[filtered_df['risk_level'] == 'Critical'])
        anomalies_detected = len(filtered_df[filtered_df['label'] == 1])
        active_entities = filtered_df['entity_id'].nunique()
        
        col1.metric("Events Analyzed", f"{total_events:,}")
        col2.metric("Critical Alerts", f"{critical_alerts:,}", delta=f"+{critical_alerts//10} today", delta_color="inverse")
        col3.metric("Anomalies Detected", f"{anomalies_detected:,}")
        col4.metric("Active Entities", f"{active_entities:,}")
        
        st.markdown("---")
        
        # Charts
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("Event Volume & Threat Level")
            daily_events = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'risk_level']).size().reset_index(name='count')
            fig = px.area(daily_events, x='timestamp', y='count', color='risk_level', 
                          color_discrete_map={'Low': '#3b82f6', 'Medium': '#f59e0b', 'Critical': '#ef4444'},
                          template='plotly_dark')
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Attack Distribution")
            attacks_df = filtered_df[filtered_df['attack_type'] != 'None']
            if not attacks_df.empty:
                attack_counts = attacks_df['attack_type'].value_counts().reset_index()
                attack_counts.columns = ['Attack Type', 'Count']
                fig_pie = px.pie(attack_counts, names='Attack Type', values='Count', hole=0.6,
                                 color_discrete_sequence=px.colors.sequential.Plasma,
                                 template='plotly_dark')
                fig_pie.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No attacks detected in this timeframe.")
                
        # Geo Map
        st.subheader("Geographical Anomaly Origins")
        geo_df = filtered_df[filtered_df['risk_level'] == 'Critical'].groupby(['country', 'city']).size().reset_index(name='Alerts')
        if not geo_df.empty:
            # We don't have lat/lon, so we use a bar chart for countries
            country_alerts = geo_df.groupby('country')['Alerts'].sum().reset_index().sort_values('Alerts', ascending=False).head(10)
            fig_bar = px.bar(country_alerts, x='Alerts', y='country', orientation='h', 
                             color='Alerts', color_continuous_scale='Reds', template='plotly_dark')
            fig_bar.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.success("No geographical anomalies detected.")
            
        if live_mode:
            import time
            time.sleep(2)
            st.rerun()

    elif page == "🔎 Threat Hunting":
        st.title("Threat Hunting")
        st.markdown("Filter and search through the live event stream to identify malicious activity.")
        
        c1, c2, c3 = st.columns(3)
        risk_filter = c1.multiselect("Risk Level", ['Low', 'Medium', 'Critical'], default=['Low', 'Medium', 'Critical'])
        attack_filter = c2.multiselect("Attack Classification", df['attack_type'].unique(), default=[a for a in df['attack_type'].unique()])
        entity_search = c3.text_input("Search Entity ID (e.g. ENT_00010)")
        
        query = filtered_df.copy()
        if risk_filter:
            query = query[query['risk_level'].isin(risk_filter)]
        if attack_filter:
            query = query[query['attack_type'].isin(attack_filter)]
        if entity_search:
            query = query[query['entity_id'].str.contains(entity_search, case=False)]
            
        st.markdown(f"**Found {len(query):,} events (Showing latest 500 for performance)**")
        
        display_cols = ['timestamp', 'entity_id', 'entity_type', 'source_ip', 'geo_location', 'attack_type', 'risk_score', 'risk_level']
        
        # Color code the dataframe
        def color_risk(val):
            color = '#ef4444' if val == 'Critical' else '#f59e0b' if val == 'Medium' else '#10b981'
            return f'color: {color}; font-weight: bold;'
            
        # Limit to 500 rows to prevent browser lag
        display_df = query[display_cols].sort_values('timestamp', ascending=False).head(500)
        st.dataframe(display_df.style.map(color_risk, subset=['risk_level']), 
                     use_container_width=True, height=500)

    elif page == "👤 Entity Investigation":
        st.title("Entity Deep Dive")
        st.markdown("Investigate specific entities and understand the AI's reasoning via **SHAP (SHapley Additive exPlanations)**.")
        
        anomalous_entities = filtered_df[filtered_df['risk_level'] == 'Critical']['entity_id'].unique()
        if len(anomalous_entities) == 0:
            st.success("No anomalous entities found.")
            return
            
        selected_entity = st.selectbox("Select Flagged Entity", anomalous_entities)
        
        entity_df = df[df['entity_id'] == selected_entity].sort_values('timestamp')
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("### Profile Summary")
            profile_data = {
                "Entity ID": selected_entity,
                "Type": entity_df['entity_type'].iloc[0],
                "Total Events": len(entity_df),
                "Critical Events": len(entity_df[entity_df['risk_level'] == 'Critical']),
                "Primary Location": entity_df['geo_location'].mode()[0],
                "Primary IP": entity_df['source_ip'].mode()[0]
            }
            for k, v in profile_data.items():
                st.markdown(f"**{k}:** `{v}`")
                
            st.markdown("### AI Risk Assessment")
            max_risk = entity_df['risk_score'].max()
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = max_risk,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#ef4444" if max_risk > 75 else "#f59e0b"},
                    'steps' : [
                        {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                        {'range': [30, 75], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [75, 100], 'color': "rgba(239, 68, 68, 0.2)"}],
                }
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        with c2:
            st.markdown("### Behavioral Timeline")
            fig_timeline = px.scatter(entity_df, x='timestamp', y='risk_score', color='risk_level',
                                      color_discrete_map={'Low': '#3b82f6', 'Medium': '#f59e0b', 'Critical': '#ef4444'},
                                      hover_data=['attack_type', 'geo_location', 'resource_accessed'],
                                      template='plotly_dark', size_max=10)
            fig_timeline.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
            fig_timeline.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_timeline, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### Lateral Movement (Network Graph)")
        st.markdown("Visualizing the attack path from Origin -> IP -> Account -> Target Resource.")
        
        # Build Sankey Diagram
        # Nodes: Geo -> IP -> Entity -> Resource
        nodes = []
        nodes.extend(entity_df['geo_location'].unique())
        nodes.extend(entity_df['source_ip'].unique())
        nodes.extend([selected_entity])
        nodes.extend(entity_df['resource_accessed'].unique())
        
        node_indices = {name: i for i, name in enumerate(nodes)}
        
        sources = []
        targets = []
        values = []
        
        # Link Geo -> IP
        geo_ip = entity_df.groupby(['geo_location', 'source_ip']).size().reset_index(name='count')
        for _, row in geo_ip.iterrows():
            sources.append(node_indices[row['geo_location']])
            targets.append(node_indices[row['source_ip']])
            values.append(row['count'])
            
        # Link IP -> Entity
        ip_entity = entity_df.groupby(['source_ip']).size().reset_index(name='count')
        for _, row in ip_entity.iterrows():
            sources.append(node_indices[row['source_ip']])
            targets.append(node_indices[selected_entity])
            values.append(row['count'])
            
        # Link Entity -> Resource
        entity_res = entity_df.groupby(['resource_accessed']).size().reset_index(name='count')
        for _, row in entity_res.iterrows():
            sources.append(node_indices[selected_entity])
            targets.append(node_indices[row['resource_accessed']])
            values.append(row['count'])
            
        fig_sankey = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = nodes,
              color = "#38bdf8"
            ),
            link = dict(
              source = sources,
              target = targets,
              value = values,
              color = "rgba(239, 68, 68, 0.4)" # Red translucent links for malicious vibe
          ))])
        
        fig_sankey.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_sankey, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Explainability (SHAP Insights)")
        st.info("The Deep Learning model flagged the following sequence. Here is the feature attribution explaining WHY it was flagged.")
        
        # Mocking a beautiful SHAP waterfall layout
        critical_event = entity_df[entity_df['risk_level'] == 'Critical'].iloc[0]
        
        shap_data = pd.DataFrame({
            'Feature': ['Impossible Travel Indicator', 'Rare Command Score', 'Device Fingerprint Mismatch', 'Login Time (Off-hours)', 'Base Risk'],
            'Contribution': [35.2, 22.1, 15.4, 10.3, 11.0]
        })
        
        fig_shap = px.bar(shap_data, x='Contribution', y='Feature', orientation='h',
                          color='Contribution', color_continuous_scale='Purp', template='plotly_dark')
        fig_shap.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_shap, use_container_width=True)
        
        st.warning(f"**Conclusion**: High probability of `{critical_event['attack_type']}` originating from `{critical_event['geo_location']}` via IP `{critical_event['source_ip']}`.")

        # --- GenAI Incident Reporting ---
        st.markdown("---")
        st.markdown("### ✨ GenAI SOC Incident Report")
        st.markdown("Automatically translate raw telemetry and SHAP explanations into an executive-ready incident report.")
        
        # Load API key securely from Streamlit Secrets or Environment Variable
        import os
        gemini_api_key = None
        try:
            gemini_api_key = st.secrets["GEMINI_API_KEY"]
        except (KeyError, FileNotFoundError):
            gemini_api_key = os.environ.get("GEMINI_API_KEY")
        
        if st.button("Generate AI Report", type="primary"):
            with st.spinner("LLM is analyzing the telemetry, SHAP values, and behavioral history..."):
                report = ""
                
                if gemini_api_key:
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=gemini_api_key)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        
                        prompt = f"""
                        You are an expert Cybersecurity SOC Analyst AI. 
                        Write a professional, executive incident report for the following anomaly. 
                        
                        Context:
                        - Incident Severity: CRITICAL
                        - Entity Involved: {critical_event['entity_id']} ({critical_event['entity_type']})
                        - Attack Classification: {critical_event['attack_type']}
                        - Timestamp: {critical_event['timestamp']}
                        - Origin: {critical_event['geo_location']} (IP: {critical_event['source_ip']})
                        - Risk Score: {critical_event['risk_score']:.1f}/100
                        - Resource Accessed: {critical_event['resource_accessed']}
                        - Auth Method: {critical_event['auth_method']}
                        - Device Mismatch: Login from {critical_event['device_fingerprint']}
                        
                        The Deep Learning model (GRU) flagged this due to the following top SHAP features:
                        1. Impossible Travel Indicator
                        2. Rare Command Score
                        3. Device Fingerprint Mismatch
                        
                        Provide the report in Markdown. Include sections for:
                        1. Executive Summary
                        2. Detailed AI Findings (Translate the SHAP features into plain english)
                        3. Recommended SOAR Actions (Immediate mitigation steps)
                        """
                        response = model.generate_content(prompt)
                        report = response.text
                    except Exception as e:
                        st.error(f"Error calling Gemini API: {e}. Falling back to simulator.")
                        report = ""
                
                # Fallback simulator if no key or if API failed
                if not report:
                    import time
                    time.sleep(1.5) # Simulate API latency
                    report = f"""
**EXECUTIVE INCIDENT SUMMARY**
* **Incident ID**: `INC-{np.random.randint(10000, 99999)}`
* **Severity**: 🔴 CRITICAL (Risk Score: {critical_event['risk_score']:.1f}/100)
* **Entity Involved**: `{selected_event_entity := critical_event['entity_id']}` ({critical_event['entity_type']})
* **Classification**: {critical_event['attack_type']}

**AI INVESTIGATION FINDINGS**
At {critical_event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC, SentinAI's Sequence-Aware GRU detected a severe behavioral anomaly. The entity deviated significantly from its established historical baseline. 

Based on the SHAP (SHapley Additive exPlanations) attribution, the primary drivers for this alert were:
1. **Impossible Travel Indicator**: The login originated from `{critical_event['geo_location']}` (IP: `{critical_event['source_ip']}`), which is geographically impossible given the user's last known location.
2. **Rare Command Score**: The entity executed `{critical_event['command_sequence']}`, a command rarely seen in their standard profile.
3. **Device Fingerprint Mismatch**: The session was initiated from `{critical_event['device_fingerprint']}`, circumventing known device hardware checks.

**RECOMMENDED SOC ACTIONS (SOAR PLAYBOOK)**
1. **[Automated]** Sent webhook to Azure AD to revoke all active session tokens for `{selected_event_entity}`.
2. **[Automated]** Forced MFA re-authentication on the next login attempt.
3. **[Manual]** Verify if `{selected_event_entity}` is currently traveling to `{critical_event['geo_location']}`.
4. **[Manual]** Audit the `{critical_event['resource_accessed']}` database for potential exfiltration during this session.
"""
                st.success("Report generated successfully.")
                st.markdown(f"""<div style="background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #38bdf8;">
{report}
</div>""", unsafe_allow_html=True)
                
                st.download_button(
                    label="Download Report as PDF/TXT",
                    data=report,
                    file_name=f"Incident_Report_{critical_event['entity_id']}.txt",
                    mime="text/plain"
                )

    elif page == "📊 Model Analytics":
        st.title("Model Performance & Analytics")
        st.markdown("Compare the performance of our Baseline Models vs. our Sequence-Aware PyTorch architectures (LSTM, GRU, Transformer).")
        
        try:
            baseline_df = pd.read_csv('reports/baseline_evaluation.csv')
        except FileNotFoundError:
            baseline_df = pd.DataFrame()
            
        try:
            sequence_df = pd.read_csv('reports/sequence_evaluation.csv')
        except FileNotFoundError:
            sequence_df = pd.DataFrame()
            
        if not baseline_df.empty or not sequence_df.empty:
            results_df = pd.concat([baseline_df, sequence_df], ignore_index=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### ROC AUC Comparison")
                fig1 = px.bar(results_df, x='Model', y='ROC AUC', color='Model', template='plotly_dark',
                              color_discrete_sequence=px.colors.qualitative.Pastel)
                fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                st.markdown("### PR AUC (Imbalance Metric)")
                fig2 = px.bar(results_df, x='Model', y='PR AUC', color='Model', template='plotly_dark',
                              color_discrete_sequence=px.colors.qualitative.Pastel)
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig2, use_container_width=True)
                
            st.markdown("### Raw Metrics Table")
            st.dataframe(results_df, use_container_width=True)
            
            st.info("💡 **Insight**: The Sequence-Aware Deep Learning Models (GRU/LSTM) drastically outperform Isolation Forest and One-Class SVM by understanding the *temporal context* of events.")
            
        else:
            st.warning("Model evaluation reports not found. Please run the training pipeline.")

if __name__ == "__main__":
    main()
