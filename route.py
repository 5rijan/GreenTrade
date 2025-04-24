import csv
import math
import re
import traceback
from collections import defaultdict
import os
import itertools

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCODE_FILENAME = os.path.join(SCRIPT_DIR, 'code_list.csv')

FUNC_POS = {
    'SeaPort': 0, 'Rail': 1, 'Road': 2, 'Airport': 3,
    'Postal': 4, 'ICD': 5, 'FixedTransport': 6, 'BorderCrossing': 7
}
CODE_SEAPORT = '1'
CODE_AIRPORT = '4'
CODE_ICD = '6'

VALID_STATUSES = {'AA', 'AC', 'AF', 'AI', 'AS', 'RL', 'RQ', 'RR', 'QQ', 'XX'}
EARTH_RADIUS_KM = 6371
NUM_NEARBY_HUBS = 3 # Hubs close to Origin/Destination
NUM_MID_HUBS_PER_POINT = 5 # Hubs near points along the route
MAX_RESULTS = 15
MID_ROUTE_SEARCH_RADIUS_KM = 1500 # Max deviation from path for mid-hubs

EMISSIONS_KG_PER_TKM = {
    'Sea':   0.016,
    'Air':   0.602,
    'Truck': 0.105,
    'Land':  0.105 # Default 
}

SPEED_PER_KM = {
    'Sea':   32,
    'Air':   85,
    'Truck': 65,
    'Land':  60 # Default 
}

TIME_AVERAGE_COEFF_PER_HUB_TYPE_IN_HOURS = {
    'Sea':   30,
    'Air':   8,
    'Truck': 6,
    'Land':  6 # Default 
}
# --- Helper Functions ---
def parse_coordinates(coord_str):
    # (Keep existing parse_coordinates function as is)
    if not coord_str or not isinstance(coord_str, str): return None
    coord_str = coord_str.strip().upper()
    lat, lon = None, None
    # DDMM[SS]N DDDMM[SS]E format
    match = re.match(r"(\d{2})(\d{2})\d{0,2}([NS])\s?(\d{3})(\d{2})\d{0,2}([EW])", coord_str)
    if match:
        try:
            d1, m1, ns, d2, m2, ew = match.groups()
            lat = int(d1) + int(m1) / 60.0
            lon = int(d2) + int(m2) / 60.0
            if ns == 'S': lat = -lat
            if ew == 'W': lon = -lon
        except ValueError:
            pass # Will try float parsing next

    # Try simple float parsing if the first format failed or wasn't present
    if lat is None or lon is None:
        # Remove common non-numeric characters except decimal point and sign
        cleaned_coord_str = re.sub(r"[^\d\.\s\-\+NSEW]", "", coord_str)
        # Split by space, comma, or semicolon, filtering empty strings
        parts = [p for p in re.split(r'[ ,;]+', cleaned_coord_str) if p]
        if len(parts) == 2:
            try:
                p1_str, p2_str = parts
                # Handle N/S/E/W suffixes if present
                lat_sign = 1.0
                lon_sign = 1.0
                if p1_str.endswith(('N', 'n')): lat_sign = 1.0; p1_str = p1_str[:-1]
                elif p1_str.endswith(('S', 's')): lat_sign = -1.0; p1_str = p1_str[:-1]
                if p2_str.endswith(('E', 'e')): lon_sign = 1.0; p2_str = p2_str[:-1]
                elif p2_str.endswith(('W', 'w')): lon_sign = -1.0; p2_str = p2_str[:-1]

                # Handle possible +/- prefixes
                if p1_str.startswith('+'): p1_str = p1_str[1:]
                if p1_str.startswith('-'): lat_sign = -1.0; p1_str = p1_str[1:]
                if p2_str.startswith('+'): p2_str = p2_str[1:]
                if p2_str.startswith('-'): lon_sign = -1.0; p2_str = p2_str[1:]

                lat_val = float(p1_str) * lat_sign
                lon_val = float(p2_str) * lon_sign

                # Basic validation: Assume standard order (Lat, Lon)
                # Could add heuristics here if order is often swapped
                if -90 <= lat_val <= 90 and -180 <= lon_val <= 180:
                    lat, lon = lat_val, lon_val
                # Check if maybe swapped (Lon, Lat)
                elif -90 <= lon_val <= 90 and -180 <= lat_val <= 180:
                     # print(f"Warning: Possible swapped coordinates '{coord_str}'. Assuming Lat={lon_val}, Lon={lat_val}.") # Optional Warning
                     lat, lon = lon_val, lat_val # Use swapped values

            except ValueError:
                 pass # Parsing failed

    if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
        # print(f"Parsed '{coord_str}' -> ({lat:.4f}, {lon:.4f})") # Optional Debug
        return lat, lon
    # print(f"Failed to parse coordinates: '{coord_str}'") # Optional Debug
    return None

def calculate_leg_travel_time(distance_km, mode):
    """Calculates travel time for a leg in hours."""
    if distance_km <= 0:
        return 0
    try:
        # Get speed, default to 'Land' speed if mode not found
        speed_kmh = SPEED_PER_KM.get(mode, SPEED_PER_KM.get('Land'))
        if speed_kmh is None or speed_kmh <= 0:
            print(f"Warning: Invalid speed ({speed_kmh}) for mode '{mode}' or fallback 'Land'. Cannot calculate travel time for leg.")
            return float('inf') # Or return 0, or handle differently? Returning inf indicates an issue. Let's try 0 for simplicity in sums.
            # return 0 # Returning 0 might hide issues but prevents breaking sums.

        # Ensure speed is positive before division
        if speed_kmh > 0:
             return distance_km / speed_kmh
        else:
             print(f"Warning: Zero or negative speed ({speed_kmh}) for mode '{mode}'. Returning 0 travel time.")
             return 0 # Avoid division by zero

    except Exception as e:
        print(f"Error calculating travel time for dist={distance_km}, mode={mode}: {e}")
        return 0 # Return 0 on error to avoid breaking totals

def get_hub_dwell_time(hub_entry):
    """Gets the estimated dwell/handling time at a hub in hours based on its type."""
    if not hub_entry:
        return 0

    # Prioritize the type based on likely handling time (Sea > Air > Land/Truck)
    if hub_entry.get('IsSeaPort'):
        mode_for_dwell = 'Sea'
    elif hub_entry.get('IsAirport'):
        mode_for_dwell = 'Air'
    else:
        # Assume 'Land' or 'Truck' if it's an intermediate point but not Sea/Air
        # Could add checks for IsRoad, IsRail, IsICD if specific times are needed
        mode_for_dwell = 'Land' # Default to Land

    # Get dwell time, default to 'Land' dwell time if specific mode not found
    dwell_time = TIME_AVERAGE_COEFF_PER_HUB_TYPE_IN_HOURS.get(
        mode_for_dwell, TIME_AVERAGE_COEFF_PER_HUB_TYPE_IN_HOURS.get('Land', 0) # Default to 0 if 'Land' somehow missing
    )
    return dwell_time if dwell_time is not None else 0


def haversine(lat1, lon1, lat2, lon2):
    # (Keep existing haversine function as is)
    if None in [lat1, lon1, lat2, lon2]: return float('inf')
    try:
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return c * EARTH_RADIUS_KM
    except ValueError:
        # print(f"ValueError in haversine: {(lat1, lon1)} -> {(lat2, lon2)}") # Debug
        return float('inf')

# --- THIS FUNCTION WAS MISSING ---
def get_closest_hubs(target_coords, hub_list, num_hubs):
    """Finds the num_hubs closest hubs from hub_list to target_coords."""
    if not target_coords: return []
    lat1, lon1 = target_coords
    with_dist = []
    for hub in hub_list:
        # Ensure hub has coordinates before calculating distance
        hub_coords = hub.get('CoordsParsed')
        if hub_coords:
            lat2, lon2 = hub_coords
            dist = haversine(lat1, lon1, lat2, lon2)
            # Ensure distance is valid and greater than a small threshold (avoid self-matching)
            if dist < float('inf') and dist > 0.1: # Use 0.1 km as a small threshold
                with_dist.append({'hub': hub, 'dist': dist})
        # else: # Optional: Log hubs skipped due to missing coords
        #     print(f"Skipping hub {hub.get('DisplayName', 'Unknown')} in get_closest_hubs due to missing CoordsParsed.")

    with_dist.sort(key=lambda x: x['dist'])
    return [item['hub'] for item in with_dist[:num_hubs]]
# --- END OF MISSING FUNCTION ---

def calculate_intermediate_point(lat1, lon1, lat2, lon2, fraction):
    """Calculates coordinates at a fraction along the great circle path."""
    if None in [lat1, lon1, lat2, lon2]: return None
    try:
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        d = 2 * math.asin(math.sqrt(math.sin((lat2-lat1)/2)**2 +
                           math.cos(lat1) * math.cos(lat2) * math.sin((lon2-lon1)/2)**2))

        if abs(d) < 1e-9 : # Check if points are identical or very close
            return math.degrees(lat1), math.degrees(lon1)

        A = math.sin((1-fraction)*d) / math.sin(d)
        B = math.sin(fraction*d) / math.sin(d)

        x = A * math.cos(lat1) * math.cos(lon1) + B * math.cos(lat2) * math.cos(lon2)
        y = A * math.cos(lat1) * math.sin(lon1) + B * math.cos(lat2) * math.sin(lon2)
        z = A * math.sin(lat1) + B * math.sin(lat2)

        lat_i = math.atan2(z, math.sqrt(x**2 + y**2))
        lon_i = math.atan2(y, x)

        return math.degrees(lat_i), math.degrees(lon_i)
    except (ValueError, ZeroDivisionError):
        # print(f"Error calculating intermediate point for {(lat1, lon1)} -> {(lat2, lon2)} frac={fraction}")
        return None

def find_hubs_near_point(target_coords, hub_list, search_radius_km, num_hubs):
    """Finds up to num_hubs from hub_list within search_radius_km of target_coords."""
    if not target_coords: return []
    lat, lon = target_coords
    with_dist = []
    for hub in hub_list:
        hub_coords = hub.get('CoordsParsed')
        if hub_coords:
            dist = haversine(lat, lon, hub_coords[0], hub_coords[1])
            if dist <= search_radius_km: # Check if within radius
                with_dist.append({'hub': hub, 'dist': dist})
    with_dist.sort(key=lambda x: x['dist'])
    return [item['hub'] for item in with_dist[:num_hubs]]


# --- LOCODE Data Loading ---
def load_locode_data(filename=LOCODE_FILENAME):
    # (Keep existing load_locode_data function as is)
    data = []
    if not os.path.exists(filename):
        print(f"ERROR: LOCODE file not found: {filename}")
        return None
    print(f"Loading LOCODE data from {filename}...")
    loaded_count = 0
    skipped_count = 0
    skipped_no_coords = 0
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header
            for i, row in enumerate(reader):
                if len(row) < 12:
                    # print(f"Warn: Row {i+1} too short ({len(row)} columns), skipping.")
                    skipped_count += 1
                    continue
                try:
                    entry = {
                        'Country': row[1].strip(),
                        'Location': row[2].strip(),
                        'Name': row[3].strip(),
                        'NameWoDiacritics': row[4].strip(),
                        'Subdivision': row[5].strip(),
                        'Status': row[6].strip().upper(),
                        'Function': row[7].strip().ljust(8, '-') if row[7] else '--------',
                        'Date': row[8].strip(),
                        'IATA': row[9].strip().upper(),
                        'Coordinates': row[10].strip(),
                        'Remarks': row[11].strip()
                    }
                    func_str = entry['Function']
                    entry['IsSeaPort'] = func_str[FUNC_POS['SeaPort']] == CODE_SEAPORT
                    entry['IsAirport'] = func_str[FUNC_POS['Airport']] == CODE_AIRPORT or \
                                        (entry['IATA'] and len(entry['IATA']) == 3 and entry['IATA'].isalpha())
                    entry['IsICD'] = func_str[FUNC_POS['ICD']] == CODE_ICD
                    entry['IsRoad'] = func_str[FUNC_POS['Road']] == '3'
                    entry['IsRail'] = func_str[FUNC_POS['Rail']] == '2'

                    # Attempt coordinate parsing immediately
                    entry['CoordsParsed'] = parse_coordinates(entry['Coordinates'])

                    # Enhanced validation: Require Coordinates for Hubs, check Status
                    is_valid_hub_type = entry['IsSeaPort'] or entry['IsAirport'] # or entry['IsICD'] # Decide if ICDs need coords
                    is_valid_status = entry['Status'] in VALID_STATUSES

                    if entry['Country'] and entry['Location'] and entry['Name'] and is_valid_status:
                        # Only count as skipped due to no coords if it *should* have them
                        if not entry['CoordsParsed'] and is_valid_hub_type:
                            # print(f"Warn: Row {i+1} ({entry.get('NameWoDiacritics', 'unknown')}, {entry.get('Country', 'unknown')}-{entry.get('Location', 'unknown')}) skipped: Missing or invalid coordinates '{entry['Coordinates']}' for Hub.")
                            skipped_no_coords += 1
                            skipped_count += 1
                            continue # Skip hubs without coordinates

                        iata = f" [{entry['IATA']}]" if entry['IsAirport'] and entry['IATA'] else ""
                        htype = "Port" if entry['IsSeaPort'] else "Airport" if entry['IsAirport'] else "ICD" if entry['IsICD'] else "Location"
                        entry['DisplayName'] = f"{entry['NameWoDiacritics']}, {entry['Country']}-{entry['Location']} ({htype}){iata}"
                        data.append(entry)
                        loaded_count += 1
                    else:
                        reason = []
                        if not entry['Country']: reason.append("no Country")
                        if not entry['Location']: reason.append("no Location")
                        if not entry['Name']: reason.append("no Name")
                        if not is_valid_status: reason.append(f"invalid Status '{entry['Status']}'")
                        # Don't print skip warning if it was already skipped for coords above
                        if not (not entry['CoordsParsed'] and is_valid_hub_type):
                             # print(f"Warn: Skipped row {i+1} ({entry.get('NameWoDiacritics', 'unknown')}, {entry.get('Country', 'unknown')}-{entry.get('Location', 'unknown')}): {', '.join(reason)}")
                             skipped_count += 1

                except Exception as e:
                    print(f"Warn: Proc. row {i+1} err: {e}")
                    skipped_count += 1
    except Exception as e:
        print(f"ERROR reading LOCODE: {e}")
        traceback.print_exc()
        return None
    print(f"Loaded {loaded_count} valid LOCODE entries (skipped {skipped_count}, incl. {skipped_no_coords} hubs missing coords).")
    if not data:
        print("WARNING: No data loaded.")
    return data


def get_location_list(locode_data):
    # (Keep existing get_location_list function as is)
    if not locode_data: return ["--- No Locations Loaded ---"]
    return sorted([e['DisplayName'] for e in locode_data if e.get('DisplayName')], key=lambda x: x.split(',')[0].lower())


def find_location_by_display_name(display_name, locode_data):
    # (Keep existing find_location_by_display_name function as is)
    if not locode_data or not display_name: return None
    # Optimize slightly: create a temporary lookup if called repeatedly, but for single calls, this is fine.
    return next((e for e in locode_data if e.get('DisplayName') == display_name), None)


def calculate_carbon(distance_km, weight_kg, mode):
    # (Keep existing calculate_carbon function as is)
    if weight_kg <= 0 or distance_km <= 0: return 0 # <<< THIS IS THE MOST LIKELY CAUSE
    try:
        factor = EMISSIONS_KG_PER_TKM.get(mode, EMISSIONS_KG_PER_TKM['Land']) # Use 'Land' as fallback
        weight_tonnes = weight_kg / 1000.0
        return round(distance_km * factor * weight_tonnes)
    except Exception:
        return 0
    
# --- Enhanced Routing Logic ---

def get_leg_mode(loc1_entry, loc2_entry):
    """Determines the most likely transport mode between two LOCODE entries."""
    if loc1_entry is None or loc2_entry is None:
        return 'Land' # Fallback

    # Check direct Sea connection
    if loc1_entry.get('IsSeaPort') and loc2_entry.get('IsSeaPort'):
        return 'Sea'

    # Check direct Air connection
    if loc1_entry.get('IsAirport') and loc2_entry.get('IsAirport'):
        return 'Air'

    # Default to Land (Truck/Rail) if no direct Sea/Air hub connection
    # More nuanced: if one is sea/air and the other isn't, it's likely land to/from hub
    return 'Land'

# Assuming these helper functions are defined ABOVE this function in route.py:
# - calculate_leg_travel_time(distance_km, mode)
# - get_hub_dwell_time(hub_entry)
# - calculate_carbon(distance_km, weight_kg, mode)
# - get_leg_mode(loc1_entry, loc2_entry)
# - haversine(lat1, lon1, lat2, lon2)
# - find_location_by_display_name(display_name, locode_data)
# - get_closest_hubs(target_coords, hub_list, num_hubs)
# - calculate_intermediate_point(lat1, lon1, lat2, lon2, fraction)
# - find_hubs_near_point(target_coords, hub_list, search_radius_km, num_hubs)
# And assuming SPEED_KMH and HUB_DWELL_TIME_HOURS dictionaries are defined globally.

def calculate_routes_from_locode(origin_entry, dest_entry, weight_kg, locode_data):
    if not all([origin_entry, dest_entry, locode_data]):
        print("Error: Missing origin, destination, or locode data.")
        return []

    print(f"\nCalculating routes: {origin_entry.get('DisplayName','Unknown Origin')} -> {dest_entry.get('DisplayName','Unknown Dest')} ({weight_kg}kg)")
    possible_routes = []
    processed_route_sigs = set() # Use to prevent adding effectively duplicate routes

    origin_coords = origin_entry.get('CoordsParsed')
    dest_coords = dest_entry.get('CoordsParsed')

    if not origin_coords:
        print(f"Warning: Missing coordinates for origin: {origin_entry.get('DisplayName')}.")
    if not dest_coords:
        print(f"Warning: Missing coordinates for destination: {dest_entry.get('DisplayName')}.")

    origin_name = origin_entry.get('NameWoDiacritics', 'Origin')
    dest_name = dest_entry.get('NameWoDiacritics', 'Destination')

    # --- Internal Helper to Add Routes ---
    def add_route_if_new(route_details):
        sig_parts = [
            route_details.get('Origin'), route_details.get('Destination'),
            route_details.get('Type'), route_details.get('Intermediate Hub 1', ''),
            route_details.get('Intermediate Hub 2', '')
        ]
        sig = tuple(p if p is not None else '' for p in sig_parts)

        if sig not in processed_route_sigs:
            # Default values (including time)
            route_details.setdefault('Origin', origin_entry.get('DisplayName'))
            route_details.setdefault('Destination', dest_entry.get('DisplayName'))
            route_details.setdefault('Details', f"Route ({weight_kg}kg)")
            route_details.setdefault('Est. Carbon (kg CO2e)', 0)
            route_details.setdefault('Est. Time (hours)', 0) # <<< Default time added
            route_details.setdefault('Route Desc', "N/A")
            route_details.setdefault('Type', "Unknown")
            route_details.setdefault('Distance', 0)
            route_details.setdefault('Legs', 0)
            route_details.setdefault('Origin_Coordinates', f"({origin_coords[0]:.4f}, {origin_coords[1]:.4f})" if origin_coords else "N/A")
            route_details.setdefault('Destination_Coordinates', f"({dest_coords[0]:.4f}, {dest_coords[1]:.4f})" if dest_coords else "N/A")

            # Add intermediate coords if hubs present (no change needed here)
            if 'Intermediate Hub 1' in route_details and route_details['Intermediate Hub 1'] and 'Intermediate_Hub1_Coordinates' not in route_details:
                 hub1_entry = find_location_by_display_name(route_details['Intermediate Hub 1'], locode_data)
                 hub1_coords = hub1_entry.get('CoordsParsed') if hub1_entry else None
                 route_details['Intermediate_Hub1_Coordinates'] = f"({hub1_coords[0]:.4f}, {hub1_coords[1]:.4f})" if hub1_coords else "N/A"
            if 'Intermediate Hub 2' in route_details and route_details['Intermediate Hub 2'] and 'Intermediate_Hub2_Coordinates' not in route_details:
                 hub2_entry = find_location_by_display_name(route_details['Intermediate Hub 2'], locode_data)
                 hub2_coords = hub2_entry.get('CoordsParsed') if hub2_entry else None
                 route_details['Intermediate_Hub2_Coordinates'] = f"({hub2_coords[0]:.4f}, {hub2_coords[1]:.4f})" if hub2_coords else "N/A"

            possible_routes.append(route_details)
            processed_route_sigs.add(sig)
    # --- End Internal Helper ---


    print("  Filtering active Sea and Air hubs with coordinates...")
    all_sea_hubs = [loc for loc in locode_data if loc.get('IsSeaPort') and loc.get('CoordsParsed') and loc.get('Status') in VALID_STATUSES]
    all_air_hubs = [loc for loc in locode_data if loc.get('IsAirport') and loc.get('CoordsParsed') and loc.get('Status') in VALID_STATUSES]
    print(f"  Found {len(all_sea_hubs)} Sea hubs, {len(all_air_hubs)} Air hubs.")


    print("  Checking Direct Routes...")
    direct_dist = float('inf') # Initialize
    if origin_coords and dest_coords:
        direct_dist = haversine(origin_coords[0], origin_coords[1], dest_coords[0], dest_coords[1])

        if direct_dist < float('inf') and direct_dist > 0.1: # Avoid zero distance routes
            # Direct Sea
            if origin_entry.get('IsSeaPort') and dest_entry.get('IsSeaPort'):
                mode = 'Sea'
                carbon = calculate_carbon(direct_dist, weight_kg, mode)
                # --- TIME CALC (Direct) ---
                travel_time_hours = calculate_leg_travel_time(direct_dist, mode)
                total_time_hours = travel_time_hours # No dwell
                # --- END TIME CALC ---
                add_route_if_new({
                    'Details': f"Direct Sea Freight ({weight_kg}kg)", 'Est. Carbon (kg CO2e)': carbon,
                    'Route Desc': f"Direct Sea: {origin_name} <> {dest_name} ({round(direct_dist)}km)",
                    'Type': 'Direct Sea', 'Distance': round(direct_dist), 'Legs': 1,
                    'Est. Time (hours)': round(total_time_hours, 1) # <<< Add time
                })
            # Direct Air
            if origin_entry.get('IsAirport') and dest_entry.get('IsAirport'):
                mode = 'Air'
                carbon = calculate_carbon(direct_dist, weight_kg, mode)
                # --- TIME CALC (Direct) ---
                travel_time_hours = calculate_leg_travel_time(direct_dist, mode)
                total_time_hours = travel_time_hours # No dwell
                # --- END TIME CALC ---
                add_route_if_new({
                    'Details': f"Direct Air Freight ({weight_kg}kg)", 'Est. Carbon (kg CO2e)': carbon,
                    'Route Desc': f"Direct Air: {origin_name} <> {dest_name} ({round(direct_dist)}km)",
                    'Type': 'Direct Air', 'Distance': round(direct_dist), 'Legs': 1,
                    'Est. Time (hours)': round(total_time_hours, 1) # <<< Add time
                })
            # Direct Land
            is_origin_land_possible = not (origin_entry.get('IsSeaPort') or origin_entry.get('IsAirport')) or origin_entry.get('IsRoad') or origin_entry.get('IsRail') or origin_entry.get('IsICD')
            is_dest_land_possible = not (dest_entry.get('IsSeaPort') or dest_entry.get('IsAirport')) or dest_entry.get('IsRoad') or dest_entry.get('IsRail') or dest_entry.get('IsICD')
            if is_origin_land_possible and is_dest_land_possible and direct_dist < 6000: # Increased limit slightly
                mode = 'Truck' # Assume Truck for direct land
                carbon = calculate_carbon(direct_dist, weight_kg, mode)
                # --- TIME CALC (Direct) ---
                travel_time_hours = calculate_leg_travel_time(direct_dist, mode)
                total_time_hours = travel_time_hours # No dwell
                # --- END TIME CALC ---
                add_route_if_new({
                    'Details': f"Direct Land Transport ({weight_kg}kg)", 'Est. Carbon (kg CO2e)': carbon,
                    'Route Desc': f"Direct Land: {origin_name} <> {dest_name} ({round(direct_dist)}km)",
                    'Type': 'Direct Land', 'Distance': round(direct_dist), 'Legs': 1,
                    'Est. Time (hours)': round(total_time_hours, 1) # <<< Add time
                })
        elif direct_dist <= 0.1 and direct_dist != float('inf'): # Check for near-zero distance
             print(f"Warning: Origin and Destination ({origin_name}, {dest_name}) are effectively the same location (dist={direct_dist}km).")
    else:
        print("  Skipping Direct Routes calculation due to missing origin/destination coordinates.")


    # --- Find Nearby Hubs ---
    print("  Finding Nearby Hubs (close to origin/destination)...")
    nearby_origin_sea_hubs = []
    nearby_origin_air_hubs = []
    nearby_dest_sea_hubs = []
    nearby_dest_air_hubs = []

    if origin_coords:
        nearby_origin_sea_hubs = get_closest_hubs(origin_coords, all_sea_hubs, NUM_NEARBY_HUBS)
        nearby_origin_air_hubs = get_closest_hubs(origin_coords, all_air_hubs, NUM_NEARBY_HUBS)
        # print(f"    Found {len(nearby_origin_sea_hubs)} nearby origin Sea hubs...") # Keep prints short
        # print(f"    Found {len(nearby_origin_air_hubs)} nearby origin Air hubs...")
    else:
        print("    Skipping nearby origin hub search (missing origin coords).")

    if dest_coords:
        nearby_dest_sea_hubs = get_closest_hubs(dest_coords, all_sea_hubs, NUM_NEARBY_HUBS)
        nearby_dest_air_hubs = get_closest_hubs(dest_coords, all_air_hubs, NUM_NEARBY_HUBS)
        # print(f"    Found {len(nearby_dest_sea_hubs)} nearby dest Sea hubs...")
        # print(f"    Found {len(nearby_dest_air_hubs)} nearby dest Air hubs...")
    else:
        print("    Skipping nearby destination hub search (missing dest coords).")


    # --- Find Mid-Route Hubs ---
    print(f"  Finding Mid-Route Hubs (near path, within {MID_ROUTE_SEARCH_RADIUS_KM}km radius)...")
    mid_route_sea_hubs = []
    mid_route_air_hubs = []
    if origin_coords and dest_coords and direct_dist > MID_ROUTE_SEARCH_RADIUS_KM * 1.2 : # Check distance
        point_33 = calculate_intermediate_point(origin_coords[0], origin_coords[1], dest_coords[0], dest_coords[1], 0.33)
        point_67 = calculate_intermediate_point(origin_coords[0], origin_coords[1], dest_coords[0], dest_coords[1], 0.67)
        found_mid_sea = set()
        found_mid_air = set()

        for p in [point_33, point_67]:
            if p:
                hubs_near_p_sea = find_hubs_near_point(p, all_sea_hubs, MID_ROUTE_SEARCH_RADIUS_KM, NUM_MID_HUBS_PER_POINT)
                for hub in hubs_near_p_sea:
                    if hub['DisplayName'] not in [origin_entry.get('DisplayName'), dest_entry.get('DisplayName')] and hub['DisplayName'] not in found_mid_sea:
                        mid_route_sea_hubs.append(hub)
                        found_mid_sea.add(hub['DisplayName'])
                hubs_near_p_air = find_hubs_near_point(p, all_air_hubs, MID_ROUTE_SEARCH_RADIUS_KM, NUM_MID_HUBS_PER_POINT)
                for hub in hubs_near_p_air:
                    if hub['DisplayName'] not in [origin_entry.get('DisplayName'), dest_entry.get('DisplayName')] and hub['DisplayName'] not in found_mid_air:
                        mid_route_air_hubs.append(hub)
                        found_mid_air.add(hub['DisplayName'])

        # print(f"    Found {len(mid_route_sea_hubs)} potential mid-route Sea hubs...")
        # print(f"    Found {len(mid_route_air_hubs)} potential mid-route Air hubs...")
    else:
        print("    Skipping Mid-Route hub search (origin/dest coords missing or distance too short).")


    # --- Calculate Indirect Routes using NEARBY Hubs ---
    print("  Calculating Indirect Routes (using Nearby Hubs)...")

    # Route: Origin -> Nearby Dest Hub -> Destination (2 Legs)
    if origin_coords and dest_coords:
        # Using Nearby Destination Sea Hubs
        for dest_hub in nearby_dest_sea_hubs:
            hub_coords = dest_hub['CoordsParsed']
            if dest_hub['DisplayName'] == origin_entry.get('DisplayName'): continue

            dist1_mode = get_leg_mode(origin_entry, dest_hub)
            dist2_mode = get_leg_mode(dest_hub, dest_entry)
            dist1 = haversine(origin_coords[0], origin_coords[1], hub_coords[0], hub_coords[1])
            dist2 = haversine(hub_coords[0], hub_coords[1], dest_coords[0], dest_coords[1])

            if dist1 < float('inf') and dist2 < float('inf') and (dist1 + dist2) > 0.1:
                carbon1 = calculate_carbon(dist1, weight_kg, dist1_mode)
                carbon2 = calculate_carbon(dist2, weight_kg, dist2_mode)
                total_dist = round(dist1 + dist2)
                total_carbon = carbon1 + carbon2
                # --- TIME CALC (2-Leg) ---
                time1_h = calculate_leg_travel_time(dist1, dist1_mode)
                time2_h = calculate_leg_travel_time(dist2, dist2_mode)
                dwell_h = get_hub_dwell_time(dest_hub) # Dwell at intermediate hub
                total_time_hours = time1_h + time2_h + dwell_h
                # --- END TIME CALC ---
                route_type = f"Indirect {dist1_mode}+{dist2_mode}"
                add_route_if_new({
                    'Details': f"Indirect via {dest_hub['NameWoDiacritics']} ({dest_hub.get('Location', 'N/A')}) ({weight_kg}kg)",
                    'Est. Carbon (kg CO2e)': total_carbon,
                    'Route Desc': f"{route_type}: {origin_name} -> {dest_hub['NameWoDiacritics']} -> {dest_name} ({total_dist}km)",
                    'Type': route_type, 'Distance': total_dist, 'Legs': 2,
                    'Intermediate Hub 1': dest_hub['DisplayName'],
                    'Intermediate_Hub1_Coordinates': f"({hub_coords[0]:.4f}, {hub_coords[1]:.4f})",
                    'Est. Time (hours)': round(total_time_hours, 1) # <<< Add time
                })
        # Using Nearby Destination Air Hubs
        for dest_hub in nearby_dest_air_hubs:
            hub_coords = dest_hub['CoordsParsed']
            if dest_hub['DisplayName'] == origin_entry.get('DisplayName'): continue

            dist1_mode = get_leg_mode(origin_entry, dest_hub)
            dist2_mode = get_leg_mode(dest_hub, dest_entry)
            dist1 = haversine(origin_coords[0], origin_coords[1], hub_coords[0], hub_coords[1])
            dist2 = haversine(hub_coords[0], hub_coords[1], dest_coords[0], dest_coords[1])

            if dist1 < float('inf') and dist2 < float('inf') and (dist1 + dist2) > 0.1:
                carbon1 = calculate_carbon(dist1, weight_kg, dist1_mode)
                carbon2 = calculate_carbon(dist2, weight_kg, dist2_mode)
                total_dist = round(dist1 + dist2)
                total_carbon = carbon1 + carbon2
                # --- TIME CALC (2-Leg) ---
                time1_h = calculate_leg_travel_time(dist1, dist1_mode)
                time2_h = calculate_leg_travel_time(dist2, dist2_mode)
                dwell_h = get_hub_dwell_time(dest_hub)
                total_time_hours = time1_h + time2_h + dwell_h
                # --- END TIME CALC ---
                route_type = f"Indirect {dist1_mode}+{dist2_mode}"
                add_route_if_new({
                    'Details': f"Indirect via {dest_hub['NameWoDiacritics']} ({dest_hub.get('Location', 'N/A')}) ({weight_kg}kg)",
                    'Est. Carbon (kg CO2e)': total_carbon,
                    'Route Desc': f"{route_type}: {origin_name} -> {dest_hub['NameWoDiacritics']} -> {dest_name} ({total_dist}km)",
                    'Type': route_type, 'Distance': total_dist, 'Legs': 2,
                    'Intermediate Hub 1': dest_hub['DisplayName'],
                    'Intermediate_Hub1_Coordinates': f"({hub_coords[0]:.4f}, {hub_coords[1]:.4f})",
                    'Est. Time (hours)': round(total_time_hours, 1) # <<< Add time
                })

        # Route: Origin -> Nearby Origin Hub -> Destination (2 Legs)
        # Using Nearby Origin Sea Hubs
        for origin_hub in nearby_origin_sea_hubs:
            hub_coords = origin_hub['CoordsParsed']
            if origin_hub['DisplayName'] == dest_entry.get('DisplayName'): continue

            dist1_mode = get_leg_mode(origin_entry, origin_hub)
            dist2_mode = get_leg_mode(origin_hub, dest_entry)
            dist1 = haversine(origin_coords[0], origin_coords[1], hub_coords[0], hub_coords[1])
            dist2 = haversine(hub_coords[0], hub_coords[1], dest_coords[0], dest_coords[1])

            if dist1 < float('inf') and dist2 < float('inf') and (dist1 + dist2) > 0.1:
                carbon1 = calculate_carbon(dist1, weight_kg, dist1_mode)
                carbon2 = calculate_carbon(dist2, weight_kg, dist2_mode)
                total_dist = round(dist1 + dist2)
                total_carbon = carbon1 + carbon2
                # --- TIME CALC (2-Leg) ---
                time1_h = calculate_leg_travel_time(dist1, dist1_mode)
                time2_h = calculate_leg_travel_time(dist2, dist2_mode)
                dwell_h = get_hub_dwell_time(origin_hub) # Dwell at intermediate hub
                total_time_hours = time1_h + time2_h + dwell_h
                # --- END TIME CALC ---
                route_type = f"Indirect {dist1_mode}+{dist2_mode}"
                add_route_if_new({
                    'Details': f"Indirect via {origin_hub['NameWoDiacritics']} ({origin_hub.get('Location', 'N/A')}) ({weight_kg}kg)",
                    'Est. Carbon (kg CO2e)': total_carbon,
                    'Route Desc': f"{route_type}: {origin_name} -> {origin_hub['NameWoDiacritics']} -> {dest_name} ({total_dist}km)",
                    'Type': route_type, 'Distance': total_dist, 'Legs': 2,
                    'Intermediate Hub 1': origin_hub['DisplayName'],
                    'Intermediate_Hub1_Coordinates': f"({hub_coords[0]:.4f}, {hub_coords[1]:.4f})",
                    'Est. Time (hours)': round(total_time_hours, 1) # <<< Add time
                })
        # Using Nearby Origin Air Hubs
        for origin_hub in nearby_origin_air_hubs:
            hub_coords = origin_hub['CoordsParsed']
            if origin_hub['DisplayName'] == dest_entry.get('DisplayName'): continue

            dist1_mode = get_leg_mode(origin_entry, origin_hub)
            dist2_mode = get_leg_mode(origin_hub, dest_entry)
            dist1 = haversine(origin_coords[0], origin_coords[1], hub_coords[0], hub_coords[1])
            dist2 = haversine(hub_coords[0], hub_coords[1], dest_coords[0], dest_coords[1])

            if dist1 < float('inf') and dist2 < float('inf') and (dist1 + dist2) > 0.1:
                carbon1 = calculate_carbon(dist1, weight_kg, dist1_mode)
                carbon2 = calculate_carbon(dist2, weight_kg, dist2_mode)
                total_dist = round(dist1 + dist2)
                total_carbon = carbon1 + carbon2
                # --- TIME CALC (2-Leg) ---
                time1_h = calculate_leg_travel_time(dist1, dist1_mode)
                time2_h = calculate_leg_travel_time(dist2, dist2_mode)
                dwell_h = get_hub_dwell_time(origin_hub)
                total_time_hours = time1_h + time2_h + dwell_h
                # --- END TIME CALC ---
                route_type = f"Indirect {dist1_mode}+{dist2_mode}"
                add_route_if_new({
                    'Details': f"Indirect via {origin_hub['NameWoDiacritics']} ({origin_hub.get('Location', 'N/A')}) ({weight_kg}kg)",
                    'Est. Carbon (kg CO2e)': total_carbon,
                    'Route Desc': f"{route_type}: {origin_name} -> {origin_hub['NameWoDiacritics']} -> {dest_name} ({total_dist}km)",
                    'Type': route_type, 'Distance': total_dist, 'Legs': 2,
                    'Intermediate Hub 1': origin_hub['DisplayName'],
                    'Intermediate_Hub1_Coordinates': f"({hub_coords[0]:.4f}, {hub_coords[1]:.4f})",
                    'Est. Time (hours)': round(total_time_hours, 1) # <<< Add time
                })

        # Route: Origin -> Nearby Origin Hub -> Nearby Dest Hub -> Destination (3 Legs)
        # Sea -> Sea -> Land/Sea
        for hub1 in nearby_origin_sea_hubs:
            hub1_coords = hub1['CoordsParsed']
            if hub1['DisplayName'] == origin_entry.get('DisplayName'): continue
            for hub2 in nearby_dest_sea_hubs:
                hub2_coords = hub2['CoordsParsed']
                if hub2['DisplayName'] == dest_entry.get('DisplayName') or hub1['DisplayName'] == hub2['DisplayName']: continue

                dist1_mode = get_leg_mode(origin_entry, hub1)
                dist2_mode = get_leg_mode(hub1, hub2)
                dist3_mode = get_leg_mode(hub2, dest_entry)
                dist1 = haversine(origin_coords[0], origin_coords[1], hub1_coords[0], hub1_coords[1])
                dist2 = haversine(hub1_coords[0], hub1_coords[1], hub2_coords[0], hub2_coords[1])
                dist3 = haversine(hub2_coords[0], hub2_coords[1], dest_coords[0], dest_coords[1])

                if dist1 < float('inf') and dist2 < float('inf') and dist3 < float('inf') and (dist1 + dist2 + dist3) > 0.1:
                    carbon1 = calculate_carbon(dist1, weight_kg, dist1_mode)
                    carbon2 = calculate_carbon(dist2, weight_kg, dist2_mode)
                    carbon3 = calculate_carbon(dist3, weight_kg, dist3_mode)
                    total_dist = round(dist1 + dist2 + dist3)
                    total_carbon = carbon1 + carbon2 + carbon3
                    # --- TIME CALC (3-Leg) ---
                    time1_h = calculate_leg_travel_time(dist1, dist1_mode)
                    time2_h = calculate_leg_travel_time(dist2, dist2_mode)
                    time3_h = calculate_leg_travel_time(dist3, dist3_mode)
                    dwell1_h = get_hub_dwell_time(hub1) # Dwell at Hub 1
                    dwell2_h = get_hub_dwell_time(hub2) # Dwell at Hub 2
                    total_time_hours = time1_h + time2_h + time3_h + dwell1_h + dwell2_h
                    # --- END TIME CALC ---
                    route_type = f"Indirect {dist1_mode}+{dist2_mode}+{dist3_mode}"
                    add_route_if_new({
                        'Details': f"Indirect via {hub1['NameWoDiacritics']} & {hub2['NameWoDiacritics']} ({weight_kg}kg)",
                        'Est. Carbon (kg CO2e)': total_carbon,
                        'Route Desc': f"{route_type}: {origin_name}->{hub1['NameWoDiacritics']}->{hub2['NameWoDiacritics']}->{dest_name} ({total_dist}km)",
                        'Type': route_type, 'Distance': total_dist, 'Legs': 3,
                        'Intermediate Hub 1': hub1['DisplayName'], 'Intermediate Hub 2': hub2['DisplayName'],
                        'Intermediate_Hub1_Coordinates': f"({hub1_coords[0]:.4f}, {hub1_coords[1]:.4f})",
                        'Intermediate_Hub2_Coordinates': f"({hub2_coords[0]:.4f}, {hub2_coords[1]:.4f})",
                        'Est. Time (hours)': round(total_time_hours, 1) # <<< Add time
                    })
        # Air -> Air -> Land/Air
        for hub1 in nearby_origin_air_hubs:
            hub1_coords = hub1['CoordsParsed']
            if hub1['DisplayName'] == origin_entry.get('DisplayName'): continue
            for hub2 in nearby_dest_air_hubs:
                hub2_coords = hub2['CoordsParsed']
                if hub2['DisplayName'] == dest_entry.get('DisplayName') or hub1['DisplayName'] == hub2['DisplayName']: continue

                dist1_mode = get_leg_mode(origin_entry, hub1)
                dist2_mode = get_leg_mode(hub1, hub2)
                dist3_mode = get_leg_mode(hub2, dest_entry)
                dist1 = haversine(origin_coords[0], origin_coords[1], hub1_coords[0], hub1_coords[1])
                dist2 = haversine(hub1_coords[0], hub1_coords[1], hub2_coords[0], hub2_coords[1])
                dist3 = haversine(hub2_coords[0], hub2_coords[1], dest_coords[0], dest_coords[1])

                if dist1 < float('inf') and dist2 < float('inf') and dist3 < float('inf') and (dist1 + dist2 + dist3) > 0.1:
                    carbon1 = calculate_carbon(dist1, weight_kg, dist1_mode)
                    carbon2 = calculate_carbon(dist2, weight_kg, dist2_mode)
                    carbon3 = calculate_carbon(dist3, weight_kg, dist3_mode)
                    total_dist = round(dist1 + dist2 + dist3)
                    total_carbon = carbon1 + carbon2 + carbon3
                    # --- TIME CALC (3-Leg) ---
                    time1_h = calculate_leg_travel_time(dist1, dist1_mode)
                    time2_h = calculate_leg_travel_time(dist2, dist2_mode)
                    time3_h = calculate_leg_travel_time(dist3, dist3_mode)
                    dwell1_h = get_hub_dwell_time(hub1)
                    dwell2_h = get_hub_dwell_time(hub2)
                    total_time_hours = time1_h + time2_h + time3_h + dwell1_h + dwell2_h
                    # --- END TIME CALC ---
                    route_type = f"Indirect {dist1_mode}+{dist2_mode}+{dist3_mode}"
                    add_route_if_new({
                        'Details': f"Indirect via {hub1['NameWoDiacritics']} & {hub2['NameWoDiacritics']} ({weight_kg}kg)",
                        'Est. Carbon (kg CO2e)': total_carbon,
                        'Route Desc': f"{route_type}: {origin_name}->{hub1['NameWoDiacritics']}->{hub2['NameWoDiacritics']}->{dest_name} ({total_dist}km)",
                        'Type': route_type, 'Distance': total_dist, 'Legs': 3,
                        'Intermediate Hub 1': hub1['DisplayName'], 'Intermediate Hub 2': hub2['DisplayName'],
                        'Intermediate_Hub1_Coordinates': f"({hub1_coords[0]:.4f}, {hub1_coords[1]:.4f})",
                        'Intermediate_Hub2_Coordinates': f"({hub2_coords[0]:.4f}, {hub2_coords[1]:.4f})",
                        'Est. Time (hours)': round(total_time_hours, 1) # <<< Add time
                    })
    else:
        print("    Skipping some nearby hub route calculations due to missing origin/destination coordinates.")


    # --- Calculate Indirect Routes using MID-ROUTE Hubs ---
    print("  Calculating Indirect Routes (using Mid-Route Hubs)...")
    if origin_coords and dest_coords:
        # Via Mid-Route Sea Hub
        for mid_hub in mid_route_sea_hubs:
            hub_coords = mid_hub['CoordsParsed']
            dist1_mode = get_leg_mode(origin_entry, mid_hub)
            dist2_mode = get_leg_mode(mid_hub, dest_entry)
            dist1 = haversine(origin_coords[0], origin_coords[1], hub_coords[0], hub_coords[1])
            dist2 = haversine(hub_coords[0], hub_coords[1], dest_coords[0], dest_coords[1])

            if dist1 < float('inf') and dist2 < float('inf') and (dist1 + dist2) > 0.1:
                carbon1 = calculate_carbon(dist1, weight_kg, dist1_mode)
                carbon2 = calculate_carbon(dist2, weight_kg, dist2_mode)
                total_dist = round(dist1 + dist2)
                total_carbon = carbon1 + carbon2
                # --- TIME CALC (Mid-Hub - treat as 2 legs + 1 dwell) ---
                time1_h = calculate_leg_travel_time(dist1, dist1_mode)
                time2_h = calculate_leg_travel_time(dist2, dist2_mode)
                dwell_h = get_hub_dwell_time(mid_hub) # Dwell at the mid hub
                total_time_hours = time1_h + time2_h + dwell_h
                # --- END TIME CALC ---
                route_type = f"Mid-Hub {dist1_mode}+{dist2_mode}"
                add_route_if_new({
                    'Details': f"Mid-Route via {mid_hub['NameWoDiacritics']} ({mid_hub.get('Location', 'N/A')}) ({weight_kg}kg)",
                    'Est. Carbon (kg CO2e)': total_carbon,
                    'Route Desc': f"{route_type}: {origin_name} -> {mid_hub['NameWoDiacritics']} -> {dest_name} ({total_dist}km)",
                    'Type': route_type, 'Distance': total_dist, 'Legs': 2, # Still count as 2 legs conceptually
                    'Intermediate Hub 1': mid_hub['DisplayName'],
                    'Intermediate_Hub1_Coordinates': f"({hub_coords[0]:.4f}, {hub_coords[1]:.4f})",
                    'Est. Time (hours)': round(total_time_hours, 1) # <<< Add time
                })

        # Via Mid-Route Air Hub
        for mid_hub in mid_route_air_hubs:
            hub_coords = mid_hub['CoordsParsed']
            dist1_mode = get_leg_mode(origin_entry, mid_hub)
            dist2_mode = get_leg_mode(mid_hub, dest_entry)
            dist1 = haversine(origin_coords[0], origin_coords[1], hub_coords[0], hub_coords[1])
            dist2 = haversine(hub_coords[0], hub_coords[1], dest_coords[0], dest_coords[1])

            if dist1 < float('inf') and dist2 < float('inf') and (dist1 + dist2) > 0.1:
                carbon1 = calculate_carbon(dist1, weight_kg, dist1_mode)
                carbon2 = calculate_carbon(dist2, weight_kg, dist2_mode)
                total_dist = round(dist1 + dist2)
                total_carbon = carbon1 + carbon2
                # --- TIME CALC (Mid-Hub - treat as 2 legs + 1 dwell) ---
                time1_h = calculate_leg_travel_time(dist1, dist1_mode)
                time2_h = calculate_leg_travel_time(dist2, dist2_mode)
                dwell_h = get_hub_dwell_time(mid_hub)
                total_time_hours = time1_h + time2_h + dwell_h
                # --- END TIME CALC ---
                route_type = f"Mid-Hub {dist1_mode}+{dist2_mode}"
                add_route_if_new({
                    'Details': f"Mid-Route via {mid_hub['NameWoDiacritics']} ({mid_hub.get('Location', 'N/A')}) ({weight_kg}kg)",
                    'Est. Carbon (kg CO2e)': total_carbon,
                    'Route Desc': f"{route_type}: {origin_name} -> {mid_hub['NameWoDiacritics']} -> {dest_name} ({total_dist}km)",
                    'Type': route_type, 'Distance': total_dist, 'Legs': 2,
                    'Intermediate Hub 1': mid_hub['DisplayName'],
                    'Intermediate_Hub1_Coordinates': f"({hub_coords[0]:.4f}, {hub_coords[1]:.4f})",
                    'Est. Time (hours)': round(total_time_hours, 1) # <<< Add time
                })
    else:
         print("    Skipping Mid-Route Hub calculations due to missing origin/destination coordinates.")


    # --- Final Sorting and Selection ---
    print(f"  Sorting {len(possible_routes)} potential routes...")
    # Sort by: Legs (fewer better), Carbon (lower better), Time (lower better), Distance (lower better)
    possible_routes.sort(key=lambda x: (
        x.get('Legs', 99),
        x.get('Est. Carbon (kg CO2e)', float('inf')),
        x.get('Est. Time (hours)', float('inf')), # <<< Sort by time
        x.get('Distance', float('inf'))
    ))

    print(f"Found {len(possible_routes)} unique routes. Returning top {min(MAX_RESULTS, len(possible_routes))}.")
    return possible_routes[:MAX_RESULTS]