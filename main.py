import pandas as pd
from fuzzywuzzy import fuzz
import shapefile
import os
import shutil

def find_line_intersection(line1, line2):
    """
    Find the intersection point of two lines.
    
    Parameters:
    -----------
    line1 : list of (x, y) tuples
        First line
    line2 : list of (x, y) tuples
        Second line
        
    Returns:
    --------
    tuple or None
        The (x, y) coordinates of the intersection point, or None if no intersection is found
    """
    # Check if lines have points
    if not line1 or not line2:
        return None
    
    # Look for intersections between each segment of both lines
    for i in range(len(line1) - 1):
        for j in range(len(line2) - 1):
            # Line segments
            p1 = line1[i]
            p2 = line1[i+1]
            p3 = line2[j]
            p4 = line2[j+1]
            
            # Line segment parameters
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            
            # Check if the line segments are not points
            if (x1 == x2 and y1 == y2) or (x3 == x4 and y3 == y4):
                continue
            
            # Calculate the denominator
            denominator = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
            
            # Lines are parallel if denominator is zero
            if denominator == 0:
                continue
            
            # Calculate parameters for the intersection point
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
            ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator
            
            # If parameters are between 0 and 1, lines intersect
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                # Calculate intersection point
                intersect_x = x1 + ua * (x2 - x1)
                intersect_y = y1 + ua * (y2 - y1)
                return (intersect_x, intersect_y)
    
    # No intersection found
    return None

def find_matched_line_intersections(lake_havasu_file, shapefile_data_file, shapefile_path, output_path, similarity_threshold=70):
    """
    Find line intersections that match criteria from the Excel file.
    
    Parameters:
    -----------
    lake_havasu_file : str
        Path to the Lake Havasu Excel file
    shapefile_data_file : str
        Path to the Shapefile Data Excel file
    shapefile_path : str
        Path to the shapefile
    output_path : str
        Path for the output point shapefile
    similarity_threshold : int
        Threshold for fuzzy matching (default: 70)
    """
    # Load Excel files
    print(f"Reading {lake_havasu_file}...")
    lake_havasu_df = pd.read_excel(lake_havasu_file)
    
    print(f"Reading {shapefile_data_file}...")
    shapefile_df = pd.read_excel(shapefile_data_file)
    
    # Verify required columns exist
    if 'Branch Name' not in lake_havasu_df.columns:
        raise ValueError("'Branch Name' column not found in Lake Havasu file")
    
    if 'StreetName' not in shapefile_df.columns:
        raise ValueError("'StreetName' column not found in Shapefile Data file")
    
    # Check for From and To columns
    from_column = None
    to_column = None
    
    # Check for common names for From/To fields
    from_candidates = ['FROM', 'From', 'FROM_NODE', 'FromNode', 'FROMADDR', 'FromAddr', 'F_ADDR', 'FADDR', 'From_', 'FROM_']
    to_candidates = ['TO', 'To', 'TO_NODE', 'ToNode', 'TOADDR', 'ToAddr', 'T_ADDR', 'TADDR', 'To_', 'TO_']
    
    for col in from_candidates:
        if col in shapefile_df.columns:
            from_column = col
            print(f"Found 'From' column: {from_column}")
            break
    
    for col in to_candidates:
        if col in shapefile_df.columns:
            to_column = col
            print(f"Found 'To' column: {to_column}")
            break
    
    # Read the shapefile
    print(f"Reading shapefile: {shapefile_path}")
    sf = shapefile.Reader(shapefile_path)
    shapes = sf.shapes()
    records = sf.records()
    fields = sf.fields
    
    print(f"Loaded {len(shapes)} features from shapefile")
    
    # Normalize names for better matching
    lake_havasu_df['Branch Name_norm'] = lake_havasu_df['Branch Name'].astype(str).str.strip().str.upper()
    shapefile_df['StreetName_norm'] = shapefile_df['StreetName'].astype(str).str.strip().str.upper()
    
    if from_column:
        shapefile_df[f'{from_column}_norm'] = shapefile_df[from_column].astype(str).str.strip().str.upper()
    
    if to_column:
        shapefile_df[f'{to_column}_norm'] = shapefile_df[to_column].astype(str).str.strip().str.upper()
    
    # Create lists to store matches
    matched_indices = set()
    match_details = []
    
    # Find matches for Branch Name to StreetName
    print(f"\nFinding matches based on Branch Name to StreetName with similarity threshold of {similarity_threshold}%...")
    
    for lake_index, lake_row in lake_havasu_df.iterrows():
        branch_name = lake_row['Branch Name_norm']
        if not isinstance(branch_name, str) or pd.isna(branch_name) or branch_name == 'NAN':
            continue
        
        for shape_index, shape_row in shapefile_df.iterrows():
            street_name = shape_row['StreetName_norm']
            if not isinstance(street_name, str) or pd.isna(street_name) or street_name == 'NAN':
                continue
            
            # Calculate similarity scores
            similarity_ratio = fuzz.ratio(branch_name, street_name)
            token_sort_ratio = fuzz.token_sort_ratio(branch_name, street_name)
            score = max(similarity_ratio, token_sort_ratio)
            
            if score >= similarity_threshold:
                matched_indices.add(shape_index)
                match_details.append({
                    'shapefile_index': shape_index,
                    'lake_havasu_index': lake_index,
                    'branch_name': lake_row['Branch Name'],
                    'street_name': shape_row['StreetName'],
                    'match_type': 'Branch Name to StreetName',
                    'score': score
                })
                print(f"Match: Branch Name '{branch_name}' to StreetName '{street_name}' with {score}% similarity")
    
    # Find matches for Branch Name to From
    if from_column:
        print(f"\nFinding matches based on Branch Name to {from_column} with similarity threshold of {similarity_threshold}%...")
        
        for lake_index, lake_row in lake_havasu_df.iterrows():
            branch_name = lake_row['Branch Name_norm']
            if not isinstance(branch_name, str) or pd.isna(branch_name) or branch_name == 'NAN':
                continue
            
            for shape_index, shape_row in shapefile_df.iterrows():
                from_value = shape_row[f'{from_column}_norm']
                if not isinstance(from_value, str) or pd.isna(from_value) or from_value == 'NAN':
                    continue
                
                # Calculate similarity scores
                similarity_ratio = fuzz.ratio(branch_name, from_value)
                token_sort_ratio = fuzz.token_sort_ratio(branch_name, from_value)
                score = max(similarity_ratio, token_sort_ratio)
                
                if score >= similarity_threshold:
                    matched_indices.add(shape_index)
                    match_details.append({
                        'shapefile_index': shape_index,
                        'lake_havasu_index': lake_index,
                        'branch_name': lake_row['Branch Name'],
                        'from_value': shape_row[from_column],
                        'match_type': 'Branch Name to From',
                        'score': score
                    })
                    print(f"Match: Branch Name '{branch_name}' to From '{from_value}' with {score}% similarity")
    
    # Find matches for Branch Name to To
    if to_column:
        print(f"\nFinding matches based on Branch Name to {to_column} with similarity threshold of {similarity_threshold}%...")
        
        for lake_index, lake_row in lake_havasu_df.iterrows():
            branch_name = lake_row['Branch Name_norm']
            if not isinstance(branch_name, str) or pd.isna(branch_name) or branch_name == 'NAN':
                continue
            
            for shape_index, shape_row in shapefile_df.iterrows():
                to_value = shape_row[f'{to_column}_norm']
                if not isinstance(to_value, str) or pd.isna(to_value) or to_value == 'NAN':
                    continue
                
                # Calculate similarity scores
                similarity_ratio = fuzz.ratio(branch_name, to_value)
                token_sort_ratio = fuzz.token_sort_ratio(branch_name, to_value)
                score = max(similarity_ratio, token_sort_ratio)
                
                if score >= similarity_threshold:
                    matched_indices.add(shape_index)
                    match_details.append({
                        'shapefile_index': shape_index,
                        'lake_havasu_index': lake_index,
                        'branch_name': lake_row['Branch Name'],
                        'to_value': shape_row[to_column],
                        'match_type': 'Branch Name to To',
                        'score': score
                    })
                    print(f"Match: Branch Name '{branch_name}' to To '{to_value}' with {score}% similarity")
    
    print(f"\nFound {len(matched_indices)} matched lines from Excel data")
    
    # Create a writer for the output shapefile
    w = shapefile.Writer(output_path)
    
    # Add fields to the output shapefile
    w.field('LINE1_IDX', 'N', 10, 0)
    w.field('LINE2_IDX', 'N', 10, 0)
    w.field('BRANCH_NAME1', 'C', 100, 0)
    w.field('BRANCH_NAME2', 'C', 100, 0)
    w.field('MATCH_TYPE1', 'C', 25, 0)
    w.field('MATCH_TYPE2', 'C', 25, 0)
    
    # Find all intersections where at least one line matches
    print("\nFinding line intersections where at least one line matches criteria...")
    intersections = []
    
    # Track progress
    total_comparisons = len(matched_indices) * len(shapes)
    progress_step = max(1, total_comparisons // 10)
    comparisons_done = 0
    
    # Dictionary to look up match details by shapefile index
    match_lookup = {}
    for match in match_details:
        idx = match['shapefile_index']
        if idx not in match_lookup:
            match_lookup[idx] = []
        match_lookup[idx].append(match)
    
    # For each matched line, check intersections with all other lines
    for i in matched_indices:
        for j in range(len(shapes)):
            # Update progress
            comparisons_done += 1
            if comparisons_done % progress_step == 0:
                progress_percent = (comparisons_done / total_comparisons) * 100
                print(f"Progress: {progress_percent:.1f}% ({comparisons_done}/{total_comparisons})")
            
            # Skip self-intersection
            if i == j:
                continue
            
            # Get shapes
            shape1 = shapes[i]
            shape2 = shapes[j]
            
            # Skip non-line shapes
            if shape1.shapeType not in [3, 5, 13, 15, 23, 25] or shape2.shapeType not in [3, 5, 13, 15, 23, 25]:
                continue
            
            # Find intersection
            intersection_point = find_line_intersection(shape1.points, shape2.points)
            
            if intersection_point:
                # Get match details for both lines (if available)
                line1_matches = match_lookup.get(i, [])
                line2_matches = match_lookup.get(j, [])
                
                # Store intersection with details
                intersection_data = {
                    'point': intersection_point,
                    'line1_idx': i,
                    'line2_idx': j,
                    'line1_matches': line1_matches,
                    'line2_matches': line2_matches
                }
                
                intersections.append(intersection_data)
                
                # Print some details
                line1_branch = line1_matches[0]['branch_name'] if line1_matches else "N/A"
                line2_branch = line2_matches[0]['branch_name'] if line2_matches else "N/A"
                print(f"Intersection found between line {i} (Branch: {line1_branch}) and line {j} (Branch: {line2_branch})")
    
    print(f"\nFound {len(intersections)} intersections involving matched lines")
    
    # Write intersections to shapefile
    for intersection in intersections:
        # Add point
        w.point(intersection['point'][0], intersection['point'][1])
        
        # Prepare record data
        line1_idx = intersection['line1_idx']
        line2_idx = intersection['line2_idx']
        
        # Get branch names
        branch_name1 = intersection['line1_matches'][0]['branch_name'] if intersection['line1_matches'] else "N/A"
        branch_name2 = intersection['line2_matches'][0]['branch_name'] if intersection['line2_matches'] else "N/A"
        
        # Get match types
        match_type1 = intersection['line1_matches'][0]['match_type'] if intersection['line1_matches'] else "N/A"
        match_type2 = intersection['line2_matches'][0]['match_type'] if intersection['line2_matches'] else "N/A"
        
        # Add record
        w.record(line1_idx, line2_idx, branch_name1, branch_name2, match_type1, match_type2)
    
    # Save shapefile
    w.close()
    
    # Copy projection file
    prj_path = shapefile_path.replace('.shp', '.prj')
    if os.path.exists(prj_path):
        try:
            output_prj = f"{output_path}.prj"
            shutil.copy2(prj_path, output_prj)
            print(f"Copied projection file to {output_prj}")
        except Exception as e:
            print(f"Warning: Could not copy projection file: {e}")
    
    print(f"Successfully created {output_path}.shp with {len(intersections)} intersection points")

if __name__ == "__main__":
    # File paths
    lake_havasu_file = "Lake Havasu Sections - Jordan.xlsx"
    shapefile_data_file = "Shapefile Data.xlsx"
    shapefile_path = "Shapefile/LakeHavasuJordan.shp"
    output_folder = "Matched_Intersections"
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    output_shapefile = os.path.join(output_folder, "LakeHavasu_Matched_Intersections")
    
    # Define similarity threshold
    similarity_threshold = 70  # Match if similarity is 70% or higher
    
    # Run the intersection finder with matches
    try:
        find_matched_line_intersections(lake_havasu_file, shapefile_data_file, shapefile_path, 
                                        output_shapefile, similarity_threshold)
    except Exception as e:
        print(f"Error: {e}")
