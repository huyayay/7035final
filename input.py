# input.py
import streamlit as st
import pandas as pd
import os
from rapidfuzz import process, fuzz
from datetime import datetime
from archive_manager import PortfolioArchive

@st.cache_data
def load_data():
    """Load all datasets"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    file_paths = {
        "factor_exposures": "Factor_Exposures.xlsx",
        "factor_cov": "Factor_Covariance_Matrix.xlsx",
        "static_data": "Static_Data.xlsx",
        "price_data": "Constituent_Price_History.csv",
        "bench_data": "benchdata.xlsx"
    }

    data = {}
    for key, filename in file_paths.items():
        filepath = os.path.join(base_dir, filename)
        try:
            if filename.endswith('.xlsx'):
                data[key] = pd.read_excel(filepath, index_col=0, engine='openpyxl')
            elif filename.endswith('.csv'):
                data[key] = pd.read_csv(filepath, parse_dates=['date'], infer_datetime_format=True)
        except Exception as e:
            st.error(f"Failed to load {filename}: {str(e)}")
            return None, None, None, None, None
    return data['factor_exposures'], data['factor_cov'], data['static_data'], data['price_data'], data['bench_data']

@st.cache_data
def get_ticker_info(static_data):
    """Generate ticker mapping table"""
    ticker_mapping = {}
    all_candidates = []
    seen = set()
    
    for _, row in static_data.iterrows():
        ticker = row['ticker'].upper()
        name = row['name'].strip().upper()
        
        if ticker not in seen:
            ticker_mapping[ticker] = row['ticker']
            all_candidates.append(ticker)
            seen.add(ticker)
        if name and name not in seen:  # Prevent empty names
            ticker_mapping[name] = row['ticker']
            all_candidates.append(name)
            seen.add(name)
    return ticker_mapping, all_candidates

def get_fuzzy_suggestions(user_input, all_candidates):
    """Real-time fuzzy search suggestions"""
    return [s[0] for s in process.extract(
        user_input, all_candidates, 
        scorer=fuzz.WRatio, 
        score_cutoff=70, 
        limit=5
    ) if s[1] >= 70]

def process_user_input():
    """Main function to process user input"""
    st.subheader("📊 Create Portfolio")
    _, _, static_data, _, _ = load_data()
    if static_data is None:
        return [], [], False

    # Initialize session state
    session = st.session_state
    session.setdefault('input_rows', [{"ticker": "", "weight": None}])
    session.setdefault('uploaded_data', None)
    session.setdefault('current_file', None)
    session.setdefault('portfolio_submitted', False)
    session.setdefault('editing_existing', None)
    
    # Get ticker mapping
    ticker_map, candidates = get_ticker_info(static_data.reset_index())
    archive = PortfolioArchive()

    # ========== File upload handling ==========
    uploaded_file = st.file_uploader(
        "Upload portfolio file (CSV/XLSX)",
        type=["csv", "xlsx"],
        help="The file should contain 'Ticker' and 'Weight' columns, with 'Weight' being a decimal between 0 and 1 (e.g., 0.15 for 15%)",
        key="portfolio_uploader"
    )
    
    # File change detection
    if uploaded_file != session.current_file:
        session.current_file = uploaded_file
        session.uploaded_data = None
        session.input_rows = [{"ticker": "", "weight": None}]

    if uploaded_file and not session.uploaded_data:
        try:
            # Read the file and keep the original weight values
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            df = df[['Ticker', 'Weight']].dropna()
            df['Ticker'] = df['Ticker'].str.upper().map(ticker_map)
            
            # Use the original weight values directly
            df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
            
            # Validate the weight range
            if df['Weight'].isnull().any() or (df['Weight'] < 0).any() or (df['Weight'] > 1).any():
                raise ValueError("Weight values must be between 0 and 1")
                
            session.uploaded_data = {
                'tickers': df['Ticker'].tolist(),
                'weights': df['Weight'].tolist()
            }
            st.success(f"✅ Successfully loaded {len(df)} stocks")
        except Exception as e:
            st.error(f"File parsing error: {str(e)}")
            st.stop()

    # ========== Manual input handling ==========
    if not session.uploaded_data:
        # Initialize original weights (edit mode)
        if session.editing_existing and 'original_weights' not in session:
            original_portfolio = next(p for p in archive.data["portfolios"] if p["name"] == session.editing_existing)
            session.original_weights = {i: w for i, w in enumerate(original_portfolio["weights"])}
        
        # Dynamically generate input rows
        for i, row in enumerate(session.input_rows):
            cols = st.columns([5, 3, 1])  # Ticker | Weight | Delete button
            with cols[0]:
                # Ticker input + real-time suggestions
                user_input = st.text_input(
                    "Ticker/Company Name",
                    value=row["ticker"],
                    key=f"ticker_{i}",
                    placeholder="e.g., AAPL or APPLE"
                ).upper()
                
                # Real-time fuzzy search
                if user_input and user_input != row.get("last_input"):
                    row["suggestions"] = [
                        s for s in candidates 
                        if user_input in s.upper()
                    ][:5]
                    row["last_input"] = user_input
                
                # Display suggestion options
                if row.get("suggestions"):
                    selected = st.selectbox(
                        "Select a match",
                        options=row["suggestions"],
                        key=f"suggest_{i}",
                        label_visibility="collapsed"
                    )
                    if selected in ticker_map:
                        row["ticker"] = ticker_map[selected]
            
            with cols[1]:
                # Weight input handling
                default_weight = row.get("weight")
                if session.editing_existing and i < len(session.original_weights):
                    default_weight = session.original_weights[i]
                
                new_weight = st.number_input(
                    "Weight",
                    value=default_weight if default_weight is not None else 0.0,
                    key=f"weight_{i}",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f",
                    help="Enter a decimal between 0 and 1 (e.g., 0.15 for 15%)"
                )
                
                # Mark the edit state
                if session.editing_existing:
                    original = session.original_weights.get(i, 0.0)
                    row["weight"] = new_weight if new_weight != original else original
                else:
                    row["weight"] = new_weight
            
            with cols[2]:
                # Delete row button (not shown for the first row)
                if i > 0 and st.button("➖", key=f"del_{i}", help="Delete this row"):
                    del session.input_rows[i]
                    st.rerun()
        
        # Action button row
        btn_cols = st.columns([4, 4, 2])  # Add Stock | Blank Area | Blank Area
        with btn_cols[0]:
            if st.button("➕ Add", help="Add a new stock input row", use_container_width=True):
                session.input_rows.append({"ticker": "", "weight": None})
                st.rerun()

    # ========== Archive management function ==========
    current_tickers, current_weights = [], []
    if session.uploaded_data:
        current_tickers = session.uploaded_data['tickers']
        current_weights = session.uploaded_data['weights']
    else:
        current_tickers = [row["ticker"] for row in session.input_rows if row["ticker"]]
        current_weights = [row["weight"] for row in session.input_rows if row["ticker"]]

    if current_tickers and current_weights:
        with st.expander("💾 Portfolio Archive Management", expanded=True):
            # Portfolio name input box
            default_name = f"Portfolio{archive.data['next_id']}" if not session.editing_existing else session.editing_existing
            custom_name = st.text_input(
                "Portfolio Name",
                value=default_name,
                help="Enter a custom portfolio name"
            )
            
            # Button row: Update and Cancel (same row layout)
            btn_cols = st.columns([4, 2, 4])  # Update | Cancel | Blank
            with btn_cols[0]:
                btn_label = "Update" if session.editing_existing else "Create Portfolio"
                if st.button(btn_label, use_container_width=True):
                    try:
                        # Automatically assign unentered weights
                        valid_entries = []
                        for ticker, weight in zip(current_tickers, current_weights):
                            if ticker in ticker_map:
                                valid_entries.append((ticker, weight))
                        
                        if not valid_entries:
                            st.error("At least one valid stock is required")
                            return [], [], False
                            
                        # Calculate entered weights and remaining weights to be assigned
                        tickers, weights = zip(*valid_entries)
                        entered_weights = [w for w in weights if w is not None]
                        remaining = max(0.0, 1.0 - sum(entered_weights))
                        unentered = len(weights) - len(entered_weights)
                        
                        # Automatically distribute the remaining weights evenly
                        if unentered > 0:
                            avg_weight = remaining / unentered
                            weights = [w if w is not None else avg_weight for w in weights]
                        
                        # Final validation
                        total_weight = sum(weights)
                        if abs(total_weight - 1.0) > 1e-6:
                            st.error(f"Abnormal total weight: {total_weight:.2%}")
                            return [], [], False
                        
                        portfolio = {
                            "name": custom_name.strip(),
                            "tickers": tickers,
                            "weights": weights,
                            "created": datetime.now().isoformat(),
                            "modified": datetime.now().isoformat()
                        }
                        # Key modification: Pass the old name for updating
                        archive.create_or_update_portfolio(
                            portfolio_data=portfolio,
                            is_update=bool(session.editing_existing),
                            old_name=session.editing_existing if session.editing_existing else None
                        )
                        st.success(f"Portfolio '{custom_name.strip()}' has been saved!")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
            
            with btn_cols[2]:
                if session.editing_existing:
                    if st.button("Cancel", use_container_width=True):
                        session.editing_existing = None
                        session.input_rows = [{"ticker": "", "weight": None}]
                        st.rerun()

    # ========== Submission handling ==========
    submit_cols = st.columns([4, 4, 2])  # Confirm Analysis | Refresh | Blank
    with submit_cols[0]:
        if st.button("Confirm", type="primary"):
            if session.uploaded_data:
                return session.uploaded_data['tickers'], session.uploaded_data['weights'], True
            
            # Automatically assign unentered weights
            valid_entries = []
            for row in session.input_rows:
                ticker = ticker_map.get(row["ticker"].upper())
                if not ticker: continue
                valid_entries.append((ticker, row["weight"]))
            
            if not valid_entries:
                st.error("At least one valid stock is required")
                return [], [], False
                
            # Calculate entered weights and remaining weights to be assigned
            tickers, weights = zip(*valid_entries)
            entered_weights = [w for w in weights if w is not None]
            remaining = max(0.0, 1.0 - sum(entered_weights))
            unentered = len(weights) - len(entered_weights)
            
            # Automatically distribute the remaining weights evenly
            if unentered > 0:
                avg_weight = remaining / unentered
                weights = [w if w is not None else avg_weight for w in weights]
            
            # Final validation
            total_weight = sum(weights)
            if abs(total_weight - 1.0) > 1e-6:
                st.error(f"Abnormal total weight: {total_weight:.2%}")
                return [], [], False
                
            return list(tickers), list(weights), True

    with submit_cols[1]:
        if st.button("🔄 Refresh", 
                type="secondary", 
                help="Clear all input content", 
                use_container_width=True):
            # Reset all input states
            session.input_rows = [{"ticker": "", "weight": None}]
            session.uploaded_data = None
            session.current_file = None
            if 'selected_portfolio' in session:
                del session.selected_portfolio
            st.rerun()

    return [], [], False