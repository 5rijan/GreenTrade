import streamlit as st
import pandas as pd
import numpy as np
import re
import pydeck as pdk
from chatbot import chatbot
import route

# --- Page Config ---
st.set_page_config(page_title="GreenTrade Companion", layout="wide", initial_sidebar_state="expanded")

# --- Initialization ---
if 'app_analyzed' not in st.session_state:
    st.session_state.app_analyzed = False
if 'selected_route_desc' not in st.session_state:
    st.session_state.selected_route_desc = None
if 'direct_route_data_df' not in st.session_state:
    st.session_state.direct_route_data_df = None

# --- Load LOCODE Data ---
@st.cache_data
def load_locode_data_cached():
    return route.load_locode_data()

@st.cache_data
def get_location_list_cached(_locode_data):
    if not _locode_data:
        return ["--- Loading Error ---"]
    valid_data = [entry for entry in _locode_data if isinstance(entry, dict) and 'DisplayName' in entry]
    location_options = sorted([entry['DisplayName'] for entry in valid_data], key=lambda x: x.split(',')[0].lower())
    location_options.insert(0, "--- Select Location ---")
    return location_options

locode_data = load_locode_data_cached()
location_list = get_location_list_cached(locode_data)

# --- Helper Function to Parse Coordinates ---
def parse_route_coordinates(coord_str):
    if not coord_str or not isinstance(coord_str, str) or coord_str == "N/A":
        return None
    match = re.match(r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)", coord_str)
    if match:
        try:
            lat, lon = map(float, match.groups())
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass
    parts = coord_str.split(',')
    if len(parts) == 2:
        try:
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass
    return None

# --- Sidebar ---
with st.sidebar:
    st.image("https://user-images.githubusercontent.com/26688034/269325638-935067ab-38ca-448c-9f68-a41f641541f1.png", width=100)
    st.markdown("### GreenTrade Companion")
    st.markdown("Enter details to find optimal import routes and view emissions.")
    st.markdown("---")
    
    st.selectbox(
        "Origin:",
        options=location_list,
        key='importing_from_select',
        index=0
    )
    st.selectbox(
        "Destination:",
        options=location_list,
        key='importing_to_select',
        index=0
    )
    st.number_input(
        "Weight (kg)",
        min_value=0.1,
        step=10.0,
        key='total_weight_input',
        value=100.0,
        format="%.1f"
    )
    if st.button("Analyze Routes", key='analyze_btn', use_container_width=True, type="primary"):
        st.session_state.app_analyzed = True
        st.session_state.selected_route_desc = None
        st.session_state.direct_route_data_df = None
        
        origin_display_name = st.session_state.get('importing_from_select')
        dest_display_name = st.session_state.get('importing_to_select')
        weight = st.session_state.get('total_weight_input', 0)
        
        if origin_display_name and origin_display_name != "--- Select Location ---" and \
           dest_display_name and dest_display_name != "--- Select Location ---" and \
           weight > 0 and locode_data:
            
            origin_entry = route.find_location_by_display_name(origin_display_name, locode_data)
            dest_entry = route.find_location_by_display_name(dest_display_name, locode_data)
            
            if origin_entry and dest_entry:
                if origin_display_name == dest_display_name:
                    st.warning("Origin and Destination cannot be the same.")
                    st.session_state.app_analyzed = False
                else:
                    with st.spinner("Calculating routes..."):
                        calculated_routes = route.calculate_routes_from_locode(
                            origin_entry, dest_entry, weight, locode_data
                        )
                        if calculated_routes:
                            st.session_state.direct_route_data_df = pd.DataFrame(calculated_routes)
                            # Set default selection to first route
                            st.session_state.selected_route_desc = st.session_state.direct_route_data_df['Route Desc'].iloc[0]
                        else:
                            st.warning("No routes found.")
                            st.session_state.direct_route_data_df = None
            else:
                st.error("Invalid origin or destination.")
                st.session_state.app_analyzed = False
        else:
            if not origin_display_name or origin_display_name == "--- Select Location ---":
                st.warning("Select a valid Origin.")
            if not dest_display_name or dest_display_name == "--- Select Location ---":
                st.warning("Select a valid Destination.")
            if not weight > 0:
                st.warning("Enter a weight greater than 0.")
            if not locode_data:
                st.error("LOCODE data failed to load.")
            st.session_state.app_analyzed = False
        
        st.rerun()

# --- Main Area ---
st.markdown("## Import Route Planner")

should_show_results = st.session_state.app_analyzed and (
    isinstance(st.session_state.get('direct_route_data_df'), pd.DataFrame) or
    st.session_state.get('direct_route_data_df') is None
)

if should_show_results:
    origin_display_name = st.session_state.get('importing_from_select')
    dest_display_name = st.session_state.get('importing_to_select')
    current_total_weight = st.session_state.get('total_weight_input', 0.0)
    
    origin_valid = origin_display_name and origin_display_name != "--- Select Location ---"
    dest_valid = dest_display_name and dest_display_name != "--- Select Location ---"
    weight_valid = current_total_weight > 0
    
    if not (origin_valid and dest_valid and weight_valid):
        st.warning("Invalid inputs. Please re-enter details and analyze.")
    else:
        col1, col2, col3 = st.columns(3)
        origin_city = origin_display_name.split(',')[0] if origin_valid else "N/A"
        dest_city = dest_display_name.split(',')[0] if dest_valid else "N/A"
        with col1:
            st.metric("Origin", origin_city)
        with col2:
            st.metric("Destination", dest_city)
        with col3:
            st.metric("Weight", f"{current_total_weight:.1f} kg")
        
        direct_route_df = st.session_state.get('direct_route_data_df')
        
        if isinstance(direct_route_df, pd.DataFrame) and not direct_route_df.empty:
            # Calculate Est. Time (days)
            if 'Est. Time (hours)' in direct_route_df.columns:
                direct_route_df['Est. Time (days)'] = direct_route_df['Est. Time (hours)'].apply(
                    lambda h: round(h / 24.0, 1) if pd.notna(h) and isinstance(h, (int, float)) and h > 0 else 0.0
                )
            
            # Define columns to display (removed Distance)
            display_cols = ['Route Desc', 'Type', 'Legs', 'Est. Carbon (kg CO2e)', 'Est. Time (days)']
            cols_to_show = [col for col in display_cols if col in direct_route_df.columns]
            
            # Sleek column configuration
            column_config_dict = {
                "Est. Carbon (kg CO2e)": st.column_config.NumberColumn(format="%d kg CO₂e"),
                "Est. Time (days)": st.column_config.NumberColumn(format="%.1f days")
            }
            
            st.markdown("### Route Options")
            st.dataframe(
                direct_route_df[cols_to_show],
                use_container_width=True,
                hide_index=True,
                column_config=column_config_dict
            )
            
            # Route selection dropdown
            direct_route_options = direct_route_df['Route Desc'].tolist()
            default_index = direct_route_options.index(st.session_state.selected_route_desc) if st.session_state.selected_route_desc in direct_route_options else 0
            
            def direct_route_select_callback():
                new_selection = st.session_state.direct_route_select_box
                st.session_state.selected_route_desc = new_selection
            
            st.selectbox(
                "Select Route:",
                options=direct_route_options,
                index=default_index,
                key='direct_route_select_box',
                on_change=direct_route_select_callback
            )
            
            # Map and Details
            if st.session_state.selected_route_desc:
                selected_direct_route_details = direct_route_df[direct_route_df['Route Desc'] == st.session_state.selected_route_desc]
                
                if not selected_direct_route_details.empty:
                    selected_route = selected_direct_route_details.iloc[0]
                    
                    st.markdown("### Route Details & Map")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Carbon Emissions", f"{selected_route.get('Est. Carbon (kg CO2e)', 'N/A')} kg CO₂e")
                    with col2:
                        time_days = selected_route.get('Est. Time (days)', 'N/A')
                        st.metric("Estimated Time", f"{time_days:.1f} days" if isinstance(time_days, (int, float)) else "N/A")
                    
                    # Map generation
                    map_points_data = []
                    arc_data = []
                    last_valid_point_coords = None
                    last_valid_point_name = None
                    
                    def add_map_point(coords, name):
                        global last_valid_point_coords, last_valid_point_name
                        if coords:
                            map_points_data.append({'lat': coords[0], 'lon': coords[1], 'name': name})
                            if last_valid_point_coords:
                                arc_data.append({
                                    "source_lat": last_valid_point_coords[0],
                                    "source_lon": last_valid_point_coords[1],
                                    "target_lat": coords[0],
                                    "target_lon": coords[1],
                                    "name": f"{last_valid_point_name} to {name}"
                                })
                            last_valid_point_coords = coords
                            last_valid_point_name = name
                    
                    origin_coords = parse_route_coordinates(selected_route.get('Origin_Coordinates'))
                    dest_coords = parse_route_coordinates(selected_route.get('Destination_Coordinates'))
                    hub1_coords = parse_route_coordinates(selected_route.get('Intermediate_Hub1_Coordinates'))
                    hub2_coords = parse_route_coordinates(selected_route.get('Intermediate_Hub2_Coordinates'))
                    
                    add_map_point(origin_coords, "Origin")
                    add_map_point(hub1_coords, "Hub 1" if hub1_coords else None)
                    add_map_point(hub2_coords, "Hub 2" if hub2_coords else None)
                    add_map_point(dest_coords, "Destination")
                    
                    if arc_data:
                        points_df = pd.DataFrame(map_points_data)
                        arcs_df = pd.DataFrame(arc_data)
                        
                        POINT_COLOR = [200, 200, 200, 220]
                        ARC_COLOR = [0, 170, 255, 190]
                        
                        scatterplot = pdk.Layer(
                            "ScatterplotLayer",
                            points_df,
                            get_position=["lon", "lat"],
                            get_color=POINT_COLOR,
                            get_radius=40000,
                            radius_min_pixels=4,
                            radius_max_pixels=15,
                            pickable=True,
                            tooltip={"html": "<b>{name}</b>"}
                        )
                        
                        arcs = pdk.Layer(
                            "ArcLayer",
                            arcs_df,
                            get_source_position=["source_lon", "source_lat"],
                            get_target_position=["target_lon", "target_lat"],
                            get_source_color=ARC_COLOR,
                            get_target_color=ARC_COLOR,
                            get_width=1.5,
                            get_tilt=15,
                            pickable=True,
                            auto_highlight=True,
                            tooltip={"html": "Leg: {name}"}
                        )
                        
                        valid_lats = [p['lat'] for p in map_points_data]
                        valid_lons = [p['lon'] for p in map_points_data]
                        mid_lat = np.mean(valid_lats) if valid_lats else 0
                        mid_lon = np.mean(valid_lons) if valid_lons else 0
                        max_lat_diff = max(valid_lats) - min(valid_lats) if len(valid_lats) > 1 else 0
                        max_lon_diff = max(valid_lons) - min(valid_lons) if len(valid_lons) > 1 else 0
                        max_diff = max(max_lat_diff, max_lon_diff)
                        zoom = 1
                        if max_diff > 0:
                            if max_diff < 5: zoom = 6
                            elif max_diff < 20: zoom = 4
                            elif max_diff < 50: zoom = 3
                            elif max_diff < 100: zoom = 2
                            else: zoom = 1
                        elif map_points_data: zoom = 5
                        
                        view_state = pdk.ViewState(
                            latitude=mid_lat,
                            longitude=mid_lon,
                            zoom=zoom,
                            pitch=40,
                            bearing=0
                        )
                        
                        st.pydeck_chart(pdk.Deck(
                            layers=[scatterplot, arcs],
                            initial_view_state=view_state,
                            map_style='mapbox://styles/mapbox/dark-v11',
                            tooltip=True
                        ))
                    elif map_points_data:
                        st.map(pd.DataFrame(map_points_data), zoom=5)
                        st.info("Route legs missing coordinates. Showing points only.")
                    else:
                        st.map(pd.DataFrame({'lat': [0], 'lon': [0]}), zoom=1)
                        st.warning("No valid coordinates for route map.")
                
                else:
                    st.error("Selected route not found. Please re-analyze.")
                    st.session_state.selected_route_desc = None
            
            # Chatbot (placed after map)
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            chatbot(data=direct_route_df)
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif st.session_state.direct_route_data_df is None:
            st.info("No routes found for the selected locations.")
    
    st.caption("Note: Routes and emissions are estimates based on standard factors. Actual values may vary.")

else:
    st.info("Enter details in the sidebar and click 'Analyze Routes' to view results.")