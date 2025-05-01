# archive_manager.py
import json
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd

ARCHIVE_PATH = Path.home() / "portfolio_archive.json"

class PortfolioArchive:
    def __init__(self):
        self.data = self._load_raw_data()
    
    def _load_raw_data(self):
        if ARCHIVE_PATH.exists():
            with open(ARCHIVE_PATH, "r") as f:
                data = json.load(f)
                for portfolio in data.get("portfolios", []):
                    portfolio.setdefault("created", datetime.now().isoformat())
                    portfolio.setdefault("modified", datetime.now().isoformat())
                return data
        return {"portfolios": [], "next_id": 1}

    def save(self):
        with open(ARCHIVE_PATH, "w") as f:
            json.dump(self.data, f, indent=2)

    def delete_portfolio(self, portfolio_name):
        self.data["portfolios"] = [
            p for p in self.data["portfolios"] 
            if p["name"] != portfolio_name
        ]
        self.save()
        return self.data

    def rename_portfolio(self, old_name, new_name):
        for p in self.data["portfolios"]:
            if p["name"] == old_name:
                p["name"] = new_name
                p["modified"] = datetime.now().isoformat()
                break
        self.save()
        return self.data

    def create_or_update_portfolio(self, portfolio_data, is_update=False, old_name=None):
        """Updated complete method"""
        portfolios = self.data["portfolios"]
        existing_names = [p["name"] for p in portfolios]

        if is_update:
            if not old_name:
                raise ValueError("Old name is required for update operation")
            if old_name not in existing_names:
                raise ValueError(f"Portfolio '{old_name}' does not exist, cannot update")
            
            new_name = portfolio_data["name"]
            # Check for name conflict (excluding itself)
            if new_name != old_name and new_name in existing_names:
                raise ValueError("Portfolio name already exists")

            # Find the old entry and update it
            index = next(i for i, p in enumerate(portfolios) if p["name"] == old_name)
            portfolios[index] = portfolio_data
        else:
            name = portfolio_data["name"]
            if name in existing_names:
                raise ValueError("Portfolio name already exists")
            if len(portfolios) >= 10:
                raise ValueError("You can save a maximum of 10 investment portfolios")
            portfolios.append(portfolio_data)
            self.data["next_id"] += 1

        self.save()
        return self.data

def show_archive_management_ui(archive):
    st.subheader("📂 Portfolio Archive Management")
    saved_portfolios = archive.data["portfolios"]
    
    # Archive clearing function
    with st.expander("⚙️ Advanced Management", expanded=False):
        # Main button for clearing operation
        if st.button("🧨 Clear All Archives",
                    help="Dangerous operation! All archives will be permanently deleted",
                    key="clear_archive_btn"):
            # Set the confirmation required status
            st.session_state.clear_confirm_needed = True
        
        # Confirmation button logic
        if st.session_state.get('clear_confirm_needed', False):
            # Wrap the confirmation button with a warning container
            with st.container(border=True):
                st.warning("Permanent Deletion Confirmation")
                cols = st.columns([2, 1, 1])
                with cols[1]:
                    # Confirmation button with a dangerous style
                    if st.button("🔥 Confirm Deletion of All Portfolios",
                                type="primary",
                                key="final_confirm_clear"):
                        # Perform the clearing operation
                        archive.data["portfolios"] = []
                        archive.data["next_id"] = 1
                        archive.save()
                        
                        # Reset all relevant session states
                        del st.session_state.clear_confirm_needed
                        if 'selected_portfolio' in st.session_state:
                            del st.session_state.selected_portfolio
                        
                        # Force page refresh
                        st.rerun()
                with cols[2]:
                    # Cancel operation
                    if st.button("↩️ Cancel"):
                        del st.session_state.clear_confirm_needed

    # Statistics
    st.metric("Current Number of Archives", f"{len(saved_portfolios)}/10")

    # Display saved portfolios and view details
    if saved_portfolios:
        for portfolio in saved_portfolios:
            with st.expander(f"📁 {portfolio['name']}", expanded=False):
                cols = st.columns([4, 2, 2, 3])
                # Display metadata
                if 'modified' in portfolio:
                    cols[0].caption(f"Modified on: {datetime.fromisoformat(portfolio['modified']).strftime('%Y-%m-%d %H:%M')}")
                
                # Button to view details
                with cols[3]:
                    if st.button(f"🔍 View Details", key=f"view_{portfolio['name']}"):
                        st.session_state.selected_portfolio = portfolio
                
                # Display detailed data
                if "selected_portfolio" in st.session_state and st.session_state.selected_portfolio["name"] == portfolio["name"]:
                    # Handle None values in the weight list
                    weights = []
                    for w in portfolio["weights"]:
                        if w is None:
                            weights.append(0.0)  # Replace None with 0.0
                        else:
                            weights.append(w)
                    df = pd.DataFrame({
                        "Stock Code": portfolio["tickers"],
                        "Weight (%)": [f"{w * 100:.2f}%" for w in weights]
                    })
                    st.dataframe(df, use_container_width=True)
                
                # Operation buttons
                with cols[1]:
                    if st.button("✏️ Edit", key=f"edit_{portfolio['name']}"):
                        st.session_state.editing_existing = portfolio['name']
                        st.session_state.input_rows = [
                            {"ticker": t, "weight": w}
                            for t, w in zip(portfolio['tickers'], portfolio['weights'])
                        ]
                        # Clear the previous original_weights
                        if 'original_weights' in st.session_state:
                            del st.session_state.original_weights
                        st.rerun()
                with cols[2]:
                    if st.button("🗑️ Delete", key=f"del_{portfolio['name']}"):
                        archive.delete_portfolio(portfolio['name'])
                        st.rerun()