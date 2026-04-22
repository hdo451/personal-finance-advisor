"""
Bank Statement Analyzer - Streamlit UI
=====================================

Beautiful web interface for your hybrid multi-agent system
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os
import copy
from dotenv import load_dotenv

try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except ImportError:
    PLOTLY_EVENTS_AVAILABLE = False

# Import your system
from main_coordinator import BankStatementAnalyzer

CATEGORY_CODES = [
    'food_dining',
    'groceries',
    'transportation',
    'shopping',
    'bills_utilities',
    'entertainment',
    'healthcare',
    'income',
    'fees',
    'other',
    'uncategorized',
]

def _category_code_to_label(code: str) -> str:
    return code.replace('_', ' ').title()

def _category_label_to_code(label: str) -> str:
    normalized = str(label).strip().lower().replace(' ', '_')
    if normalized in CATEGORY_CODES:
        return normalized
    return 'other'

def _transactions_to_editor_df(transactions: list) -> pd.DataFrame:
    """Create editable DataFrame for manual category review."""
    rows = []
    for idx, txn in enumerate(transactions):
        rows.append({
            '_txn_index': idx,
            'Date': txn['date'],
            'Description': txn['description'],
            'Category': _category_code_to_label(txn['category']),
            'Amount': txn['amount'],
            'Type': 'OUT' if txn['is_debit'] else 'IN',
            'Confidence': f"{txn['confidence']:.0%}",
            'Source': txn['source'].title()
        })
    return pd.DataFrame(rows)

def _get_category_items_for_modal(transactions: list, category_label: str) -> list:
    """Return debit transactions for the selected category, preserving table order."""
    category_code = _category_label_to_code(category_label)
    category_items = []

    for txn in transactions:
        if txn['is_debit'] and txn['category'] == category_code:
            category_items.append({
                'Description': txn['description'],
                'Amount': txn['amount']
            })

    return category_items

if hasattr(st, 'dialog'):
    @st.dialog("Category Items")
    def _show_category_items_modal(category_label: str, transactions: list):
        st.subheader(f"{category_label} - Items")
        items = _get_category_items_for_modal(transactions, category_label)

        if not items:
            st.info("No spending items found for this category.")
            return

        for item in items:
            amount_formatted = f"${item['Amount']:.2f}"
            st.write(f"{item['Description']} | {amount_formatted}")
else:
    def _show_category_items_modal(category_label: str, transactions: list):
        st.warning("Your Streamlit version does not support modal dialogs. Please upgrade Streamlit to use this feature.")

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'analyzer' not in st.session_state:
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            with st.spinner("🏗️ Initializing 3-agent system..."):
                st.session_state.analyzer = BankStatementAnalyzer(api_key)
            st.success("✅ System ready!")
        else:
            st.session_state.analyzer = None
            st.error("❌ Please set OPENAI_API_KEY in .env file")

    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    if 'generate_ai_insights' not in st.session_state:
        st.session_state.generate_ai_insights = False

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Bank Statement Analyzer",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session
    initialize_session_state()
    
    # Main header
    st.title("🏦 Bank Statement Analyzer")
    st.subheader("Hybrid Multi-Agent Financial Analysis System")
    
    # Educational warning
    st.warning("⚠️ **Educational Project Only** - Do not use with real financial data containing sensitive information")
    
    # Sidebar - System Information
    with st.sidebar:
        st.header("🤖 System Status")
        
        if st.session_state.analyzer:
            st.success("✅ All agents initialized")
            
            # Agent information
            st.subheader("Agent Architecture")
            st.write("🏗️ **Agent 1**: Document Processor")
            st.caption("Extracts & parses PDF (0 LLM calls)")
            
            st.write("🧠 **Agent 2**: Content Analyzer") 
            st.caption("Smart categorization (1 LLM call)")
            
            st.write("📊 **Agent 3**: Analysis Generator")
            st.caption("Financial insights (0-1 LLM calls)")
            
            st.divider()
            
            # Project info
            st.subheader("Project Details")
            st.write("**Course**: AIasesor, finantial AI advisor")
            st.write("**Type**: Hybrid Multi-Agent System")
            st.caption("All responses must be validated by experts.")
            st.caption("Artifitial intellillence may not be accurate.")
            st.write("**Efficiency**: ≤2 LLM calls per analysis")
            
        else:
            st.error("❌ System initialization failed")
            st.write("Please check your .env file contains OPENAI_API_KEY")
    
    # Main content area
    if not st.session_state.analyzer:
        st.error("System not initialized. Please check your API key configuration.")
        return
    
    # File upload section
    st.header("📄 Upload Bank Statement")
    
    uploaded_file = st.file_uploader(
        "Choose a bank statement PDF file",
        type=['pdf'],
        help="Upload your bank statement PDF for automated analysis"
    )
    
    if uploaded_file:
        # Show file info
        st.info(f"📄 **File**: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        
        # Analysis options
        col1, col2 = st.columns(2)
        
        with col1:
            generate_insights = st.checkbox(
                "🤖 Generate AI Insights",
                value=False,
                help="Use additional LLM call for personalized financial recommendations"
            )
        
        with col2:
            if generate_insights:
                st.info("📊 Will use 2 LLM calls (~$0.004)")
            else:
                st.info("📊 Will use 1 LLM call (~$0.002)")
        
        # Analysis button
        if st.button("🚀 Analyze Statement", type="primary", use_container_width=True):
            process_uploaded_file(uploaded_file, generate_insights)

        if st.session_state.analysis_result:
            show_debug_diagnostics(st.session_state.analysis_result)
            display_results(st.session_state.analysis_result)
    
    else:
        # Instructions when no file uploaded
        st.info("👆 Please upload a bank statement PDF to begin analysis")
        
        # Sample files section
        st.subheader("📋 Sample Files Available")
        st.write("You can test with these sample bank statements:")
        
        sample_files = [
            "chase_statement.pdf - Chase Bank format",
            "Sample Bank Statement.pdf - Generic format", 
            "Wells Fargo Statement.pdf - Wells Fargo format"
        ]
        
        for sample in sample_files:
            st.write(f"• {sample}")

def process_uploaded_file(uploaded_file, generate_insights: bool):
    """Process the uploaded file and show results"""
    
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    
    try:
        # Write uploaded file to disk
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Create processing progress
        progress_container = st.container()
        
        with progress_container:
            st.subheader("🔄 Processing Pipeline")
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Agent processing steps
            agent_cols = st.columns(3)
            
            with agent_cols[0]:
                agent1_status = st.empty()
                agent1_status.info("🏗️ Agent 1: Waiting...")
            
            with agent_cols[1]:
                agent2_status = st.empty()
                agent2_status.info("🧠 Agent 2: Waiting...")
            
            with agent_cols[2]:
                agent3_status = st.empty()
                agent3_status.info("📊 Agent 3: Waiting...")
            
            # Step 1: Agent 1
            status_text.text("🏗️ Agent 1: Processing PDF...")
            agent1_status.warning("🏗️ Agent 1: Processing PDF...")
            progress_bar.progress(20)
            
            # Step 2: Agent 2
            status_text.text("🧠 Agent 2: Smart categorization...")
            agent1_status.success("🏗️ Agent 1: ✅ Complete")
            agent2_status.warning("🧠 Agent 2: Categorizing...")
            progress_bar.progress(60)
            
            # Step 3: Agent 3
            if generate_insights:
                status_text.text("📊 Agent 3: Generating AI insights...")
            else:
                status_text.text("📊 Agent 3: Generating analysis...")
            
            agent2_status.success("🧠 Agent 2: ✅ Complete") 
            agent3_status.warning("📊 Agent 3: Analyzing...")
            progress_bar.progress(90)
            
            # Run the actual analysis
            result = st.session_state.analyzer.analyze_statement(
                temp_path,
                generate_ai_insights=generate_insights
            )
            
            # Complete
            agent3_status.success("📊 Agent 3: ✅ Complete")
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
        
        # Display results
        if result['success']:
            st.session_state.analysis_result = result
            st.session_state.generate_ai_insights = generate_insights
            st.success("🎉 Analysis completed successfully!")
        else:
            st.error(f"❌ Analysis failed: {result['error']}")
            st.info("Tip: If this is a local bank statement/cartola, try a cleaner PDF export (text-based, not scanned image).")
            show_debug_diagnostics(result)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def display_results(result: dict):
    """Display beautiful analysis results"""
    
    # Extract data
    analysis = result['analysis']
    summary = analysis['financial_summary']
    categories = analysis['category_breakdown']
    metrics = result['system_metrics']
    
    st.header("📊 Financial Analysis Results")
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Total Spent",
            f"${summary['total_spent']:.2f}",
            delta=f"{summary['debit_count']} transactions"
        )
    
    with col2:
        st.metric(
            "💵 Total Income", 
            f"${summary['total_income']:.2f}",
            delta=f"{summary['credit_count']} deposits"
        )
    
    with col3:
        net_change = summary['net_change']
        st.metric(
            "📊 Net Change",
            f"${net_change:.2f}",
            delta="Positive" if net_change > 0 else "Negative",
            delta_color="normal" if net_change > 0 else "inverse"
        )
    
    with col4:
        st.metric(
            "🤖 System Efficiency",
            f"{metrics['total_llm_calls']}/2 LLM calls",
            delta=f"${metrics['estimated_cost']:.4f} cost"
        )
    
    # Charts section
    if categories:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("🏷️ Spending by Category")
            
            # Build the pie from raw debit transactions so each slice reflects total spend.
            spending_totals = {}
            for txn in result['transactions']:
                if txn['is_debit']:
                    category_label = txn['category'].replace('_', ' ').title()
                    spending_totals[category_label] = spending_totals.get(category_label, 0.0) + float(txn['amount'])

            spending_rows = (
                pd.DataFrame(
                    [{
                        'Category': category,
                        'Amount': float(amount)
                    } for category, amount in spending_totals.items()]
                )
                .sort_values('Amount', ascending=False)
                .reset_index(drop=True)
            )

            category_names = spending_rows['Category'].tolist()
            category_amounts = spending_rows['Amount'].tolist()
            pie_colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold
            
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=category_names,
                        values=category_amounts,
                        sort=False,
                        direction='clockwise',
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(
                            colors=pie_colors[:len(category_names)],
                            line=dict(color='white', width=2)
                        )
                    )
                ]
            )
            fig_pie.update_layout(
                title="Spending Distribution",
                showlegend=True,
                margin=dict(t=40, l=10, r=10, b=10)
            )
            if PLOTLY_EVENTS_AVAILABLE:
                selected_points = plotly_events(
                    fig_pie,
                    click_event=True,
                    hover_event=False,
                    select_event=False,
                    key='spending_category_pie_events'
                )

                if selected_points:
                    point_index = selected_points[0].get('pointNumber')
                    if point_index is not None and 0 <= point_index < len(category_names):
                        _show_category_items_modal(category_names[point_index], result['transactions'])
            else:
                st.plotly_chart(fig_pie, use_container_width=True)
                st.caption("Install 'streamlit-plotly-events' to enable click-to-open category modal.")
        
        with chart_col2:
            st.subheader("📈 Category Breakdown")
            
            # Create bar chart
            df_categories = pd.DataFrame([
                {
                    'Category': cat['category'].replace('_', ' ').title(),
                    'Amount': cat['total'],
                    'Percentage': cat['percentage'],
                    'Count': cat['transaction_count']
                }
                for cat in categories[:6]  # Top 6 categories
            ])
            
            fig_bar = px.bar(
                df_categories,
                x='Category',
                y='Amount', 
                title="Top Categories by Amount",
                text='Amount',
                color='Percentage',
                color_continuous_scale='Viridis'
            )
            fig_bar.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    # Insights sections
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.subheader("💡 Financial Insights")
        for insight in analysis['basic_insights']:
            st.write(f"• {insight}")
    
    with insight_col2:
        if analysis.get('ai_insights') and analysis['ai_insights']:
            st.subheader("🤖 AI Recommendations")
            for insight in analysis['ai_insights'][:5]:  # Limit to 5
                if insight.strip():  # Only show non-empty insights
                    st.write(f"• {insight}")
        else:
            st.info("🤖 Enable 'Generate AI Insights' for personalized recommendations")
    
    # Transaction details (expandable)
    with st.expander("📋 View All Transactions", expanded=False):
        if result['transactions']:
            st.caption("You can edit categories for any row. Then click 'Update report' to recalculate metrics, charts, and LLM insights.")

            df_transactions = _transactions_to_editor_df(result['transactions'])
            category_labels = [_category_code_to_label(code) for code in CATEGORY_CODES if code != 'uncategorized']

            edited_df = st.data_editor(
                df_transactions,
                use_container_width=True,
                hide_index=True,
                key='transactions_editor',
                column_config={
                    '_txn_index': None,
                    'Amount': st.column_config.NumberColumn('Amount', format='$%.2f', disabled=True),
                    'Date': st.column_config.TextColumn('Date', disabled=True),
                    'Description': st.column_config.TextColumn('Description', disabled=True),
                    'Type': st.column_config.TextColumn('Type', disabled=True),
                    'Confidence': st.column_config.TextColumn('Confidence', disabled=True),
                    'Source': st.column_config.TextColumn('Source', disabled=True),
                    'Category': st.column_config.SelectboxColumn('Category', options=category_labels, required=True)
                }
            )

            if st.button("🔄 Update report with manual categories", type="secondary", use_container_width=True):
                _apply_manual_category_updates(result, edited_df)


def _apply_manual_category_updates(result: dict, edited_df: pd.DataFrame):
    """Apply manual category edits, persist learned rules, and recompute full analysis."""
    updated_transactions = copy.deepcopy(result['transactions'])
    changed_rows = 0
    rules_saved = 0

    for _, row in edited_df.iterrows():
        txn_index = int(row['_txn_index'])
        new_category = _category_label_to_code(row['Category'])
        old_category = updated_transactions[txn_index]['category']

        updated_transactions[txn_index]['category'] = new_category

        if new_category != old_category:
            changed_rows += 1
            saved_ok = st.session_state.analyzer.agent1.merchant_db.save_user_category_rule(
                updated_transactions[txn_index]['description'],
                new_category
            )
            if saved_ok:
                rules_saved += 1

    llm_before_calls = st.session_state.analyzer.llm.call_count
    llm_before_cost = st.session_state.analyzer.llm.total_cost

    # Reset Agent 3 counter only for this refresh run.
    st.session_state.analyzer.agent3.llm_calls_made = 0
    updated_analysis = st.session_state.analyzer.agent3.process(
        updated_transactions,
        generate_ai_insights=st.session_state.generate_ai_insights
    )

    llm_added_calls = st.session_state.analyzer.llm.call_count - llm_before_calls
    llm_added_cost = st.session_state.analyzer.llm.total_cost - llm_before_cost

    updated_result = copy.deepcopy(result)
    updated_result['transactions'] = updated_transactions
    updated_result['analysis'] = updated_analysis
    updated_result['system_metrics']['total_llm_calls'] += max(llm_added_calls, 0)
    updated_result['system_metrics']['estimated_cost'] += max(llm_added_cost, 0.0)
    updated_result['system_metrics']['processing_time'] = datetime.now().isoformat()
    updated_result['system_metrics']['agent_breakdown']['agent3_llm_calls'] += max(llm_added_calls, 0)

    st.session_state.analysis_result = updated_result

    st.success(
        f"Updated report with {changed_rows} manual category change(s). "
        f"Saved {rules_saved} learned rule(s) for future statements."
    )
    if llm_added_calls > 0:
        st.info(f"LLM refresh used {llm_added_calls} additional call(s), estimated +${llm_added_cost:.4f}.")

    st.rerun()


def show_debug_diagnostics(result: dict):
    """Show parser diagnostics to help troubleshoot unsupported PDF formats."""
    debug_data = result.get('document_debug') or result.get('debug_info') or {}

    parsing_stats = debug_data.get('parsing_stats') or debug_data.get('document_processing', {}).get('parsing_stats')
    lines = debug_data.get('sample_transaction_lines') or debug_data.get('document_processing', {}).get('raw_transaction_lines', [])[:10]

    if not parsing_stats and not lines:
        return

    with st.expander("🛠️ Parser Diagnostics", expanded=False):
        if parsing_stats:
            st.write("**Parsing stats**")
            st.json(parsing_stats)

        if lines:
            st.write("**Detected transaction-like lines (sample)**")
            for line in lines:
                st.code(line)

if __name__ == "__main__":
    main()