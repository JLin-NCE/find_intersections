import geopandas as gpd
import pandas as pd
import os
import sys
from shapely.geometry import Point
import datetime
from fuzzywuzzy import fuzz, process
import warnings

# Set the Excel file path directly in the script
# Change this to match your Excel file name
EXCEL_FILE = "Lake Havasu Sections - Jordan.xlsx"
SHAPEFILE_PATH = "Shapefile/LakeHavasuJordan.shp"
OUTPUT_DIR = "Road_Analysis_Output"

def analyze_road_sections(excel_file=EXCEL_FILE, output_file=None):
    """
    Analyze road sections, finding consecutive sections and single sections.
    Returns the analysis results.
    """
    print(f"Reading Excel file: {excel_file}")
    
    # Read Excel file
    try:
        excel_data = pd.read_excel(excel_file)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None, [], []
    
    # Check if Excel file has required columns
    if "From" not in excel_data.columns or "To" not in excel_data.columns:
        print("ERROR: Excel file must contain 'From' and 'To' columns")
        print(f"Available columns: {excel_data.columns.tolist()}")
        return None, [], []
    
    print(f"Successfully read Excel file with {len(excel_data)} rows")
    
    # Create section ID column if it doesn't exist
    if "Section_ID" not in excel_data.columns:
        # Look for potential section ID columns
        potential_id_cols = [col for col in excel_data.columns if "id" in col.lower() or "code" in col.lower() or "section" in col.lower()]
        if potential_id_cols:
            section_id_col = potential_id_cols[0]
            print(f"Using '{section_id_col}' column as Section ID")
            excel_data["Section_ID"] = excel_data[section_id_col]
        else:
            # Create a section ID based on row index
            print("No section ID column found, creating one based on row numbers")
            excel_data["Section_ID"] = excel_data.index.map(lambda x: f"Section_{x+1}")
    
    # Find consecutive sections
    consecutive_groups = []
    
    # Sort data to help find consecutive sections more effectively
    road_col = next((col for col in excel_data.columns if "road" in col.lower()), None)
    if road_col:
        print(f"Sorting data by '{road_col}' and 'From' columns")
        excel_data = excel_data.sort_values(by=[road_col, "From"])
    
    # Dictionary to store all streets that match a From to make lookup faster
    from_to_dict = {}
    for i, row in excel_data.iterrows():
        from_street = row["From"]
        if from_street not in from_to_dict:
            from_to_dict[from_street] = []
        from_to_dict[from_street].append(row.to_dict())
    
    # Find consecutive sections by checking each row
    processed_rows = set()  # Keep track of rows we've already assigned to groups
    
    print("Finding consecutive road sections...")
    for i, row in excel_data.iterrows():
        if i in processed_rows:
            continue  # Skip rows already in groups
        
        # Start a new group with this row
        current_group = [row.to_dict()]
        processed_rows.add(i)
        
        # Find next connected section(s)
        current_to = row["To"]
        while current_to:
            # Find rows where From equals current To
            next_sections = []
            if current_to in from_to_dict:
                for r in from_to_dict[current_to]:
                    try:
                        idx = excel_data.index[excel_data["Section_ID"] == r["Section_ID"]].tolist()[0]
                        if idx not in processed_rows:
                            next_sections.append(r)
                    except (IndexError, KeyError):
                        continue
            
            if not next_sections:
                break  # No more connected sections
                
            # Add the first matching section to the group
            next_section = next_sections[0]
            current_group.append(next_section)
            
            # Mark this row as processed
            try:
                next_idx = excel_data.index[excel_data["Section_ID"] == next_section["Section_ID"]].tolist()[0]
                processed_rows.add(next_idx)
            except (IndexError, KeyError):
                pass
            
            # Update current_to for the next iteration
            current_to = next_section["To"]
        
        # Save all groups (including single section groups)
        consecutive_groups.append(current_group)
    
    # Separate single-section and multi-section groups
    single_section_groups = [group for group in consecutive_groups if len(group) == 1]
    multi_section_groups = [group for group in consecutive_groups if len(group) > 1]
    
    print(f"Found {len(multi_section_groups)} consecutive section groups")
    print(f"Found {len(single_section_groups)} single sections")
    
    return excel_data, multi_section_groups, single_section_groups


def find_intersections_by_road_type(shapefile_path, excel_path, output_dir, fuzzy_threshold=80):
    """
    Find intersections in a shapefile based on consecutive or single road sections from Excel.
    Uses fuzzy string matching to handle variations in street names.
    Differentiate between consecutive and single sections.
    
    Args:
        shapefile_path (str): Path to the line shapefile
        excel_path (str): Path to the Excel file containing road sections
        output_dir (str): Directory to save the output files
        fuzzy_threshold (int): Threshold for fuzzy string matching (0-100)
    
    Returns:
        str: Path to the created shapefile or None if no intersections found
    """
    print(f"\nStarting intersection analysis...")
    print(f"Reading shapefile: {shapefile_path}")
    
    # First, analyze the Excel file to identify consecutive and single sections
    excel_data, consecutive_groups, single_sections = analyze_road_sections(excel_path)
    
    if not excel_data is not None:
        print("Could not analyze Excel data. Exiting.")
        return None
    
    # Read the shapefile
    try:
        line_gdf = gpd.read_file(shapefile_path)
        print(f"Successfully loaded shapefile with {len(line_gdf)} features")
        print(f"Shapefile columns: {line_gdf.columns.tolist()}")
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        return None
    
    # Check if StreetName exists in the shapefile
    if 'StreetName' not in line_gdf.columns:
        print("StreetName column not found in shapefile. Available columns:")
        print(line_gdf.columns.tolist())
        return None
    
    # Get all unique street names from shapefile for fuzzy matching
    unique_street_names = line_gdf['StreetName'].dropna().unique().tolist()
    print(f"Found {len(unique_street_names)} unique street names in shapefile")
    
    # Create empty lists to store intersection points and their attributes
    intersection_points = []
    intersection_attributes = []
    
    # Create a mapping for fuzzy matches to avoid repetitive calculations
    fuzzy_match_cache = {}
    
    def get_fuzzy_matches(search_term, choices, threshold):
        """Helper function to get fuzzy matches with caching"""
        if search_term in fuzzy_match_cache:
            return fuzzy_match_cache[search_term]
        
        # Get the best matches above the threshold
        matches = []
        for choice in choices:
            score = fuzz.ratio(search_term.lower(), choice.lower())
            if score >= threshold:
                matches.append((choice, score))
        
        # Sort by score (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Cache the result
        fuzzy_match_cache[search_term] = matches
        
        return matches
    
    # Process consecutive sections
    print("\nProcessing consecutive section groups...")
    for group_idx, group in enumerate(consecutive_groups):
        print(f"Processing group {group_idx+1} with {len(group)} sections")
        
        # For consecutive groups, we want to find intersection points at each 
        # transition between sections (where To of one section meets From of next)
        for section_idx in range(len(group) - 1):
            current_section = group[section_idx]
            next_section = group[section_idx + 1]
            
            # The current To should match the next From
            to_street = current_section['To']
            from_street = next_section['From']
            
            if to_street != from_street:
                print(f"  Warning: Mismatch in consecutive sections! {to_street} ≠ {from_street}")
                continue
            
            print(f"  Processing connection point: {to_street} (from {current_section['From']} to {next_section['To']})")
            
            # Get fuzzy matches for the connection street
            street_matches = get_fuzzy_matches(to_street, unique_street_names, fuzzy_threshold)
            
            if not street_matches:
                print(f"  Warning: No streets found matching '{to_street}' with threshold {fuzzy_threshold}")
                continue
            
            # Process all matches
            for street_match, match_score in street_matches:
                print(f"    Using match: '{street_match}' (score: {match_score})")
                
                # Get all lines for this matched street
                street_lines = line_gdf[line_gdf['StreetName'] == street_match]
                
                if len(street_lines) == 0:
                    print(f"    Warning: No lines found with street name '{street_match}'")
                    continue
                
                # For consecutive sections, we want to find locations where the streets meet
                # Find lines that intersect with From and To streets
                from_prev_matches = get_fuzzy_matches(current_section['From'], unique_street_names, fuzzy_threshold)
                to_next_matches = get_fuzzy_matches(next_section['To'], unique_street_names, fuzzy_threshold)
                
                if not from_prev_matches:
                    print(f"    Warning: No streets found matching '{current_section['From']}' with threshold {fuzzy_threshold}")
                    continue
                    
                if not to_next_matches:
                    print(f"    Warning: No streets found matching '{next_section['To']}' with threshold {fuzzy_threshold}")
                    continue
                
                # Get the best match for the previous From street
                prev_from_match, prev_from_score = from_prev_matches[0]
                prev_from_lines = line_gdf[line_gdf['StreetName'] == prev_from_match]
                
                # Get the best match for the next To street
                next_to_match, next_to_score = to_next_matches[0]
                next_to_lines = line_gdf[line_gdf['StreetName'] == next_to_match]
                
                # Find intersections with the connecting street
                for i, from_line in prev_from_lines.iterrows():
                    for j, connecting_line in street_lines.iterrows():
                        if from_line.geometry.intersects(connecting_line.geometry):
                            # Get the intersection point
                            intersection = from_line.geometry.intersection(connecting_line.geometry)
                            
                            # Common attributes
                            common_attrs = {
                                'RoadType': 'Consecutive',
                                'GroupID': f"Group_{group_idx+1}",
                                'NumSections': len(group),
                                'SectionNum': section_idx + 1,
                                'TotalSections': len(group),
                                'FromStreet': current_section['From'],
                                'ConnectionStreet': to_street,
                                'ToStreet': next_section['To'],
                                'FromMatch': prev_from_match,
                                'ConnectionMatch': street_match,
                                'ToMatch': next_to_match,
                                'FromScore': prev_from_score,
                                'ConnectionScore': match_score,
                                'ToScore': next_to_score,
                                'FromSectionID': current_section.get('Section_ID', f"Section_{section_idx+1}"),
                                'ToSectionID': next_section.get('Section_ID', f"Section_{section_idx+2}"),
                                'ProcessTimestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'Notes': f"Connection point in consecutive sections group {group_idx+1} (sections {section_idx+1}-{section_idx+2})",
                                'PointType': 'Connection'
                            }
                            
                            # Based on intersection geometry type
                            if intersection.geom_type == 'Point':
                                common_attrs['GeometryType'] = 'Point'
                                intersection_points.append(intersection)
                                intersection_attributes.append(common_attrs)
                            
                            elif intersection.geom_type == 'MultiPoint':
                                # Use the first point for simplicity
                                common_attrs['GeometryType'] = 'MultiPoint'
                                common_attrs['Notes'] += f" (using 1st of {len(intersection.geoms)} points)"
                                intersection_points.append(intersection.geoms[0])
                                intersection_attributes.append(common_attrs)
                            
                            else:
                                # For LineString, MultiLineString, etc., use the centroid
                                common_attrs['GeometryType'] = intersection.geom_type
                                common_attrs['Notes'] += f" (using centroid of {intersection.geom_type})"
                                intersection_points.append(intersection.centroid)
                                intersection_attributes.append(common_attrs)
                
                # Also find intersections with the next To street
                for i, connecting_line in street_lines.iterrows():
                    for j, to_line in next_to_lines.iterrows():
                        if connecting_line.geometry.intersects(to_line.geometry):
                            # Get the intersection point
                            intersection = connecting_line.geometry.intersection(to_line.geometry)
                            
                            # Common attributes
                            common_attrs = {
                                'RoadType': 'Consecutive',
                                'GroupID': f"Group_{group_idx+1}",
                                'NumSections': len(group),
                                'SectionNum': section_idx + 1,
                                'TotalSections': len(group),
                                'FromStreet': current_section['From'],
                                'ConnectionStreet': to_street,
                                'ToStreet': next_section['To'],
                                'FromMatch': prev_from_match,
                                'ConnectionMatch': street_match,
                                'ToMatch': next_to_match,
                                'FromScore': prev_from_score,
                                'ConnectionScore': match_score,
                                'ToScore': next_to_score,
                                'FromSectionID': current_section.get('Section_ID', f"Section_{section_idx+1}"),
                                'ToSectionID': next_section.get('Section_ID', f"Section_{section_idx+2}"),
                                'ProcessTimestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'Notes': f"Connection point in consecutive sections group {group_idx+1} (sections {section_idx+1}-{section_idx+2})",
                                'PointType': 'Connection'
                            }
                            
                            # Based on intersection geometry type
                            if intersection.geom_type == 'Point':
                                common_attrs['GeometryType'] = 'Point'
                                intersection_points.append(intersection)
                                intersection_attributes.append(common_attrs)
                            
                            elif intersection.geom_type == 'MultiPoint':
                                # Use the first point for simplicity
                                common_attrs['GeometryType'] = 'MultiPoint'
                                common_attrs['Notes'] += f" (using 1st of {len(intersection.geoms)} points)"
                                intersection_points.append(intersection.geoms[0])
                                intersection_attributes.append(common_attrs)
                            
                            else:
                                # For LineString, MultiLineString, etc., use the centroid
                                common_attrs['GeometryType'] = intersection.geom_type
                                common_attrs['Notes'] += f" (using centroid of {intersection.geom_type})"
                                intersection_points.append(intersection.centroid)
                                intersection_attributes.append(common_attrs)
    
    # Process single sections
    print("\nProcessing single sections...")
    for section_idx, section_group in enumerate(single_sections):
        section = section_group[0]  # Get the single section
        from_street = section['From']
        to_street = section['To']
        
        print(f"Processing single section {section_idx+1}: From={from_street}, To={to_street}")
        
        # Get fuzzy matches for From street
        from_matches = get_fuzzy_matches(from_street, unique_street_names, fuzzy_threshold)
        
        # Get fuzzy matches for To street
        to_matches = get_fuzzy_matches(to_street, unique_street_names, fuzzy_threshold)
        
        if not from_matches:
            print(f"Warning: No streets found matching 'From' value: {from_street} with threshold {fuzzy_threshold}")
            continue
            
        if not to_matches:
            print(f"Warning: No streets found matching 'To' value: {to_street} with threshold {fuzzy_threshold}")
            continue
        
        # Process all combinations of matches
        for from_street_match, from_score in from_matches[:1]:  # Just take the best match
            # Get all lines for this matching street name
            from_lines = line_gdf[line_gdf['StreetName'] == from_street_match]
            
            for to_street_match, to_score in to_matches[:1]:  # Just take the best match
                # Skip if From and To are the same street (after fuzzy matching)
                if from_street_match == to_street_match:
                    print(f"Skipping self-intersection for {from_street_match}")
                    continue
                
                # Get all lines for this matching street name
                to_lines = line_gdf[line_gdf['StreetName'] == to_street_match]
                
                print(f"  Checking intersection between '{from_street_match}' (match score: {from_score}) and '{to_street_match}' (match score: {to_score})")
                
                # Find intersections between the From and To lines
                for i, from_line in from_lines.iterrows():
                    for j, to_line in to_lines.iterrows():
                        # Check if the lines intersect
                        if from_line.geometry.intersects(to_line.geometry):
                            # Get the intersection point
                            intersection = from_line.geometry.intersection(to_line.geometry)
                            
                            # Common attributes for the single section
                            common_attrs = {
                                'RoadType': 'Single',
                                'GroupID': f"Single_{section_idx+1}",
                                'NumSections': 1,
                                'FromStreet': from_street,
                                'ToStreet': to_street,
                                'FromMatch': from_street_match,
                                'ToMatch': to_street_match,
                                'FromScore': from_score,
                                'ToScore': to_score,
                                'SectionID': section.get('Section_ID', f"Section_{section_idx+1}"),
                                'ProcessTimestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'Notes': f"Intersection for single section {section_idx+1}",
                                'PointType': 'Intersection'
                            }
                            
                            # Handle different geometry types
                            if intersection.geom_type == 'Point':
                                common_attrs['GeometryType'] = 'Point'
                                intersection_points.append(intersection)
                                intersection_attributes.append(common_attrs)
                            
                            elif intersection.geom_type == 'MultiPoint':
                                # Use the first point for simplicity
                                common_attrs['GeometryType'] = 'MultiPoint'
                                common_attrs['Notes'] += f" (using 1st of {len(intersection.geoms)} points)"
                                intersection_points.append(intersection.geoms[0])
                                intersection_attributes.append(common_attrs)
                            
                            else:
                                # For LineString, MultiLineString, etc., use the centroid
                                common_attrs['GeometryType'] = intersection.geom_type
                                common_attrs['Notes'] += f" (using centroid of {intersection.geom_type})"
                                intersection_points.append(intersection.centroid)
                                intersection_attributes.append(common_attrs)
    
    # Create a GeoDataFrame from the intersection points
    if not intersection_points:
        print("No intersections found based on the given conditions")
        return None
    
    print(f"Creating GeoDataFrame with {len(intersection_points)} intersection points")
    
    intersections_gdf = gpd.GeoDataFrame(
        intersection_attributes,
        geometry=intersection_points,
        crs=line_gdf.crs
    )
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Output shapefile path
    output_path = os.path.join(output_dir, 'Road_Intersections.shp')
    
    # Save to shapefile
    print(f"Saving {len(intersections_gdf)} intersection points to: {output_path}")
    
    # Save to GeoJSON to preserve all attribute data
    geojson_path = os.path.join(output_dir, 'Road_Intersections.geojson')
    intersections_gdf.to_file(geojson_path, driver='GeoJSON')
    print(f"Full attribute data saved to GeoJSON: {geojson_path}")
    
    # Save to shapefile (note that some attributes might be truncated)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        intersections_gdf.to_file(output_path)
    
    # Also save a CSV with all the attributes
    csv_path = os.path.join(output_dir, 'Road_Intersections_Attributes.csv')
    intersections_gdf.drop(columns='geometry').to_csv(csv_path, index=False)
    print(f"All attributes saved to CSV: {csv_path}")
    
    # Also create an Excel file with statistics
    create_summary_excel(excel_data, consecutive_groups, single_sections, intersections_gdf, output_dir)
    
    return output_path

def create_summary_excel(excel_data, consecutive_groups, single_sections, intersections_gdf, output_dir):
    """
    Create a summary Excel file with statistics about the road sections and intersections.
    """
    output_file = os.path.join(output_dir, 'Road_Analysis_Summary.xlsx')
    
    # Create consecutive sections report
    consecutive_report_rows = []
    for i, group in enumerate(consecutive_groups):
        group_row = {
            "Group_ID": f"Group_{i+1}",
            "Num_Sections": len(group),
            "Start_From": group[0]["From"],
            "End_To": group[-1]["To"],
            "Sections": ", ".join([str(section.get("Section_ID", "Unknown")) for section in group])
        }
        
        # Add more details if available
        road_name_col = next((col for col in group[0].keys() if "road" in col.lower() and "name" in col.lower()), None)
        if road_name_col and road_name_col in group[0]:
            group_row["Road_Name"] = group[0][road_name_col]
        
        consecutive_report_rows.append(group_row)
    
    # Sort by number of sections (descending)
    consecutive_report_rows.sort(key=lambda x: x["Num_Sections"], reverse=True)
    consecutive_df = pd.DataFrame(consecutive_report_rows) if consecutive_report_rows else pd.DataFrame()
    
    # Create single sections report
    single_section_rows = []
    for i, group in enumerate(single_sections):
        section = group[0]
        row = {
            "Section_ID": section.get("Section_ID", f"Unknown_{i}"),
            "From": section["From"],
            "To": section["To"]
        }
        
        # Add more details if available
        road_name_col = next((col for col in section.keys() if "road" in col.lower() and "name" in col.lower()), None)
        if road_name_col and road_name_col in section:
            row["Road_Name"] = section[road_name_col]
        
        single_section_rows.append(row)
    
    single_section_df = pd.DataFrame(single_section_rows) if single_section_rows else pd.DataFrame()
    
    # Create intersections summary
    intersection_summary = {}
    
    # Count by RoadType
    if 'RoadType' in intersections_gdf.columns:
        road_type_counts = intersections_gdf['RoadType'].value_counts().to_dict()
        for road_type, count in road_type_counts.items():
            intersection_summary[f"{road_type}_Intersections"] = count
    
    # Count by GeometryType if available
    if 'GeometryType' in intersections_gdf.columns:
        geom_type_counts = intersections_gdf['GeometryType'].value_counts().to_dict()
        for geom_type, count in geom_type_counts.items():
            intersection_summary[f"{geom_type}_Geometries"] = count
    
    # Summary statistics
    summary_data = {
        "Category": ["Total Road Sections", "Consecutive Section Groups", "Single Sections",
                    "Total Intersection Points"],
        "Count": [
            len(excel_data),
            len(consecutive_groups),
            len(single_sections),
            len(intersections_gdf)
        ],
        "Percentage": [
            "100%",
            f"{len(consecutive_groups)/len(excel_data)*100:.1f}%",
            f"{len(single_sections)/len(excel_data)*100:.1f}%",
            f"{len(intersections_gdf)/len(excel_data)*100:.1f}%"
        ]
    }
    
    # Add intersection counts by type
    for key, value in intersection_summary.items():
        summary_data["Category"].append(key)
        summary_data["Count"].append(value)
        summary_data["Percentage"].append(f"{value/len(intersections_gdf)*100:.1f}%")
    
    if consecutive_groups:
        longest_group = max(consecutive_groups, key=len)
        summary_data["Category"].append("Longest Consecutive Chain")
        summary_data["Count"].append(len(longest_group))
        summary_data["Percentage"].append(f"{len(longest_group)/len(excel_data)*100:.1f}%")
        
        summary_data["Category"].append("Longest Chain From-To")
        summary_data["Count"].append(f"{longest_group[0]['From']} to {longest_group[-1]['To']}")
        summary_data["Percentage"].append("")
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save all reports to an Excel file with multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        if not consecutive_df.empty:
            consecutive_df.to_excel(writer, sheet_name='Consecutive Sections', index=False)
        
        if not single_section_df.empty:
            single_section_df.to_excel(writer, sheet_name='Single Sections', index=False)
        
        # Save intersection statistics
        if not intersections_gdf.empty:
            # Create a simplified version for Excel (avoid geometry column)
            simple_intersections = intersections_gdf.drop(columns='geometry').copy()
            
            # Limit number of columns to avoid Excel limitations
            max_cols = 250  # Excel has a limit of 256 columns
            if len(simple_intersections.columns) > max_cols:
                simple_intersections = simple_intersections.iloc[:, :max_cols]
                print(f"Warning: Truncated intersection data to {max_cols} columns for Excel compatibility")
            
            simple_intersections.to_excel(writer, sheet_name='Intersections', index=False)
        
        # Copy the original data to a sheet as well
        excel_data.to_excel(writer, sheet_name='Original Data', index=False)
    
    print(f"Created summary Excel file at: {output_file}")

if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find intersections based on road types
    output_path = find_intersections_by_road_type(SHAPEFILE_PATH, EXCEL_FILE, OUTPUT_DIR, fuzzy_threshold=80)
    
    if output_path:
        print("\nAnalysis complete! Output files saved to:", OUTPUT_DIR)
        print("  - Shapefile:", output_path)
        print("  - GeoJSON:", os.path.join(OUTPUT_DIR, 'Road_Intersections.geojson'))
        print("  - CSV:", os.path.join(OUTPUT_DIR, 'Road_Intersections_Attributes.csv'))
        print("  - Excel:", os.path.join(OUTPUT_DIR, 'Road_Analysis_Summary.xlsx'))
    else:
        print("\nAnalysis complete but no intersections were found.")
