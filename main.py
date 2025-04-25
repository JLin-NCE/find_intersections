import pandas as pd
import shapefile
import os
import shutil
import math
from collections import defaultdict

def find_line_intersection(line1, line2):
    """
    Find the intersection point of two lines.
    
    Parameters:
    -----------
    line1 : list of (x, y) or (x, y, z) tuples
        First line
    line2 : list of (x, y) or (x, y, z) tuples
        Second line
        
    Returns:
    --------
    tuple or None
        The (x, y) or (x, y, z) coordinates of the intersection point, or None if no intersection is found
    """
    # Check if lines have points
    if not line1 or not line2:
        return None
    
    # Determine if we're dealing with 3D points
    has_z = len(line1[0]) > 2 and len(line2[0]) > 2
    
    # Look for intersections between each segment of both lines
    for i in range(len(line1) - 1):
        for j in range(len(line2) - 1):
            # Line segments
            p1 = line1[i]
            p2 = line1[i+1]
            p3 = line2[j]
            p4 = line2[j+1]
            
            # Line segment parameters (xy only)
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            x3, y3 = p3[0], p3[1]
            x4, y4 = p4[0], p4[1]
            
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
                # Calculate intersection point (xy)
                intersect_x = x1 + ua * (x2 - x1)
                intersect_y = y1 + ua * (y2 - y1)
                
                # If we have Z values, interpolate Z at intersection point
                if has_z:
                    # Get Z values
                    z1 = p1[2]
                    z2 = p2[2]
                    z3 = p3[2]
                    z4 = p4[2]
                    
                    # Interpolate Z along first line
                    intersect_z1 = z1 + ua * (z2 - z1)
                    
                    # Interpolate Z along second line
                    intersect_z2 = z3 + ub * (z4 - z3)
                    
                    # Average the two Z values (may need a more sophisticated approach)
                    intersect_z = (intersect_z1 + intersect_z2) / 2
                    
                    return (intersect_x, intersect_y, intersect_z, i, ua)
                else:
                    return (intersect_x, intersect_y, 0, i, ua)
    
    # No intersection found
    return None

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def interpolate_z(p1, p2, ratio):
    """Interpolate Z value at a specific position between two points."""
    if len(p1) > 2 and len(p2) > 2:
        z1, z2 = p1[2], p2[2]
        return z1 + ratio * (z2 - z1)
    return 0

def locate_point_on_line(point, line_points):
    """
    Find the segment index and parameter where a point falls on a line.
    
    Parameters:
    -----------
    point : tuple
        The point coordinates (x, y) or (x, y, z)
    line_points : list of tuples
        List of line vertices
    
    Returns:
    --------
    tuple
        (segment_index, parameter, distance_to_line)
        segment_index: Index of the segment this point falls on
        parameter: 0-1 value indicating position on the segment
        distance_to_line: Distance from point to line
    """
    min_dist = float('inf')
    best_segment = -1
    best_param = 0
    
    for i in range(len(line_points) - 1):
        p1 = line_points[i][:2]  # XY only
        p2 = line_points[i+1][:2]
        
        # Vector from p1 to p2
        v = (p2[0] - p1[0], p2[1] - p1[1])
        # Length squared of segment
        l2 = v[0]**2 + v[1]**2
        
        if l2 == 0:  # If segment is a point
            continue
            
        # Vector from p1 to point
        w = (point[0] - p1[0], point[1] - p1[1])
        
        # Projection of w onto v
        proj = (w[0]*v[0] + w[1]*v[1]) / l2
        
        # Clamp to segment
        proj = max(0, min(1, proj))
        
        # Closest point on segment
        closest = (p1[0] + proj * v[0], p1[1] + proj * v[1])
        
        # Distance to closest point
        dist = math.sqrt((point[0] - closest[0])**2 + (point[1] - closest[1])**2)
        
        if dist < min_dist:
            min_dist = dist
            best_segment = i
            best_param = proj
    
    return best_segment, best_param, min_dist

def split_line_at_intersections(line_points, intersections, has_z=False):
    """
    Split a line at multiple intersection points.
    
    Parameters:
    -----------
    line_points : list of tuples
        The vertices of the line
    intersections : list of tuples
        List of intersection points, each with (x, y, z, segment_index, parameter)
    has_z : bool
        Whether the line has Z values
    
    Returns:
    --------
    list of line segments
        Each segment is a list of points
    """
    if not intersections:
        return [line_points]  # Return the original line if no intersections
    
    # Create a list of all points along the line, including intersections
    all_points = []
    
    # Add the original line points with their position along the line
    cumulative_length = 0
    segment_lengths = []
    
    for i in range(len(line_points)):
        if i > 0:
            seg_length = distance(line_points[i-1][:2], line_points[i][:2])
            cumulative_length += seg_length
            segment_lengths.append(seg_length)
        
        point = line_points[i]
        all_points.append((point, cumulative_length, i, 'original'))
    
    # Add the intersection points with their position along the line
    for intersection in intersections:
        x, y, z, seg_idx, param = intersection
        
        # Calculate position along the line
        position = sum(segment_lengths[:seg_idx]) + param * segment_lengths[seg_idx]
        
        all_points.append(((x, y, z), position, seg_idx, 'intersection'))
    
    # Sort all points by their position along the line
    all_points.sort(key=lambda p: p[1])
    
    # Create segments
    segments = []
    current_segment = []
    
    for point_info in all_points:
        point, _, _, point_type = point_info
        
        # Add point to current segment
        current_segment.append(point)
        
        # If this is an intersection point, end the current segment and start a new one
        if point_type == 'intersection' and len(current_segment) > 1:
            segments.append(current_segment)
            current_segment = [point]  # Start new segment with the intersection point
    
    # Add the last segment if it has more than one point
    if len(current_segment) > 1:
        segments.append(current_segment)
    
    return segments

def find_branch_name_segments(lake_havasu_file, shapefile_data_file, shapefile_path, output_path, intersections_output):
    """
    Find segments in a shapefile that match branch names in an Excel file,
    create intersection points, and split lines at intersections.
    
    Parameters:
    -----------
    lake_havasu_file : str
        Path to the Lake Havasu Excel file
    shapefile_data_file : str
        Path to the Shapefile Data Excel file
    shapefile_path : str
        Path to the shapefile
    output_path : str
        Path for the output shapefile with split lines
    intersections_output : str
        Path for the output intersection points shapefile
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
    
    # Check if shapefile has Z values
    has_z = any(hasattr(shape, 'z') and shape.z for shape in shapes)
    if has_z:
        print("Shapefile contains Z values (elevation data)")
    
    # Determine the shape type
    shape_type = sf.shapeType
    print(f"Shapefile shape type: {shape_type}")
    
    # Normalize names for exact matching
    lake_havasu_df['Branch Name_norm'] = lake_havasu_df['Branch Name'].astype(str).str.strip().str.upper()
    shapefile_df['StreetName_norm'] = shapefile_df['StreetName'].astype(str).str.strip().str.upper()
    
    if from_column:
        shapefile_df[f'{from_column}_norm'] = shapefile_df[from_column].astype(str).str.strip().str.upper()
    
    if to_column:
        shapefile_df[f'{to_column}_norm'] = shapefile_df[to_column].astype(str).str.strip().str.upper()
    
    # Create dictionaries to store matches by shapefile index
    streetname_matches = {}
    from_matches = {}
    to_matches = {}
    all_matches = {}
    
    # Find exact matches for Branch Name to StreetName
    print("\nFinding exact matches between Branch Name and StreetName...")
    
    for lake_index, lake_row in lake_havasu_df.iterrows():
        branch_name = lake_row['Branch Name_norm']
        if not isinstance(branch_name, str) or pd.isna(branch_name) or branch_name == 'NAN':
            continue
        
        for shape_index, shape_row in shapefile_df.iterrows():
            street_name = shape_row['StreetName_norm']
            if not isinstance(street_name, str) or pd.isna(street_name) or street_name == 'NAN':
                continue
            
            # Check for exact match
            if branch_name == street_name:
                if shape_index not in streetname_matches:
                    streetname_matches[shape_index] = []
                
                # Store match details
                match_info = {
                    'lake_havasu_index': lake_index,
                    'branch_name': lake_row['Branch Name'],
                    'match_type': 'Branch Name to StreetName',
                    'match_street': True,
                    'match_from': False,
                    'match_to': False,
                    'excel_row': lake_index
                }
                
                streetname_matches[shape_index].append(match_info)
                
                # Add to all matches
                if shape_index not in all_matches:
                    all_matches[shape_index] = []
                
                all_matches[shape_index].append(match_info)
                
                print(f"Exact match: Branch Name '{branch_name}' = StreetName '{street_name}'")
    
    # Find exact matches for Branch Name to From
    if from_column:
        print(f"\nFinding exact matches between Branch Name and {from_column}...")
        
        for lake_index, lake_row in lake_havasu_df.iterrows():
            branch_name = lake_row['Branch Name_norm']
            if not isinstance(branch_name, str) or pd.isna(branch_name) or branch_name == 'NAN':
                continue
            
            for shape_index, shape_row in shapefile_df.iterrows():
                from_value = shape_row[f'{from_column}_norm']
                if not isinstance(from_value, str) or pd.isna(from_value) or from_value == 'NAN':
                    continue
                
                # Check for exact match
                if branch_name == from_value:
                    if shape_index not in from_matches:
                        from_matches[shape_index] = []
                    
                    # Store match details
                    match_info = {
                        'lake_havasu_index': lake_index,
                        'branch_name': lake_row['Branch Name'],
                        'match_type': f'Branch Name to {from_column}',
                        'match_street': False,
                        'match_from': True,
                        'match_to': False,
                        'excel_row': lake_index
                    }
                    
                    from_matches[shape_index].append(match_info)
                    
                    # Add to all matches
                    if shape_index not in all_matches:
                        all_matches[shape_index] = []
                    
                    all_matches[shape_index].append(match_info)
                    
                    print(f"Exact match: Branch Name '{branch_name}' = {from_column} '{from_value}'")
    
    # Find exact matches for Branch Name to To
    if to_column:
        print(f"\nFinding exact matches between Branch Name and {to_column}...")
        
        for lake_index, lake_row in lake_havasu_df.iterrows():
            branch_name = lake_row['Branch Name_norm']
            if not isinstance(branch_name, str) or pd.isna(branch_name) or branch_name == 'NAN':
                continue
            
            for shape_index, shape_row in shapefile_df.iterrows():
                to_value = shape_row[f'{to_column}_norm']
                if not isinstance(to_value, str) or pd.isna(to_value) or to_value == 'NAN':
                    continue
                
                # Check for exact match
                if branch_name == to_value:
                    if shape_index not in to_matches:
                        to_matches[shape_index] = []
                    
                    # Store match details
                    match_info = {
                        'lake_havasu_index': lake_index,
                        'branch_name': lake_row['Branch Name'],
                        'match_type': f'Branch Name to {to_column}',
                        'match_street': False,
                        'match_from': False,
                        'match_to': True,
                        'excel_row': lake_index
                    }
                    
                    to_matches[shape_index].append(match_info)
                    
                    # Add to all matches
                    if shape_index not in all_matches:
                        all_matches[shape_index] = []
                    
                    all_matches[shape_index].append(match_info)
                    
                    print(f"Exact match: Branch Name '{branch_name}' = {to_column} '{to_value}'")
    
    # Get matched line indices
    matched_indices = list(all_matches.keys())
    print(f"\nFound {len(matched_indices)} lines with exact matches from Excel data")
    
    # Create a writer for the intersections shapefile
    # If original has Z values, create a Z-aware shapefile
    if has_z:
        w_intersect = shapefile.Writer(intersections_output, shapeType=shapefile.POINTZ)
    else:
        w_intersect = shapefile.Writer(intersections_output)
    
    # Add fields to the intersections shapefile
    w_intersect.field('LINE1_IDX', 'N', 10, 0)
    w_intersect.field('LINE2_IDX', 'N', 10, 0)
    w_intersect.field('BRANCH1', 'C', 100, 0)
    w_intersect.field('BRANCH2', 'C', 100, 0)
    
    # Add specific match type fields
    w_intersect.field('L1_STREET', 'C', 1, 0)
    w_intersect.field('L1_FROM', 'C', 1, 0)
    w_intersect.field('L1_TO', 'C', 1, 0)
    w_intersect.field('L2_STREET', 'C', 1, 0)
    w_intersect.field('L2_FROM', 'C', 1, 0)
    w_intersect.field('L2_TO', 'C', 1, 0)
    
    # Find intersections ONLY between matched lines (both must be matched)
    print("\nFinding intersections between exact-matched lines only...")
    all_intersections = {}  # Dictionary to store intersections by line index
    intersections_list = []  # List to store all intersections for the point shapefile
    
    # Track progress
    total_comparisons = len(matched_indices) * (len(matched_indices) - 1) // 2
    progress_step = max(1, total_comparisons // 10)
    comparisons_done = 0
    
    # Check all pairs of matched lines
    for i in range(len(matched_indices)):
        for j in range(i + 1, len(matched_indices)):
            # Update progress
            comparisons_done += 1
            if total_comparisons >= 10 and comparisons_done % progress_step == 0:
                progress_percent = (comparisons_done / total_comparisons) * 100
                print(f"Progress: {progress_percent:.1f}% ({comparisons_done}/{total_comparisons})")
            
            # Get actual indices
            idx1 = matched_indices[i]
            idx2 = matched_indices[j]
            
            # Get shapes
            shape1 = shapes[idx1]
            shape2 = shapes[idx2]
            
            # Skip non-line shapes
            if shape1.shapeType not in [3, 5, 13, 15, 23, 25] or shape2.shapeType not in [3, 5, 13, 15, 23, 25]:
                continue
            
            # Prepare 3D points if available
            points1 = shape1.points
            points2 = shape2.points
            
            # If shapes have Z values, combine XY with Z
            if has_z and hasattr(shape1, 'z') and hasattr(shape2, 'z'):
                points1 = [(p[0], p[1], z) for p, z in zip(shape1.points, shape1.z)]
                points2 = [(p[0], p[1], z) for p, z in zip(shape2.points, shape2.z)]
            
            # Find intersection
            intersection = find_line_intersection(points1, points2)
            
            if intersection:
                x, y, z, seg_idx, param = intersection
                
                # Get match details
                line1_match = all_matches[idx1][0]  # Using first match if multiple exist
                line2_match = all_matches[idx2][0]  # Using first match if multiple exist
                
                # Store intersection for the point shapefile
                intersection_data = {
                    'point': (x, y, z),
                    'line1_idx': idx1,
                    'line2_idx': idx2,
                    'branch1': line1_match['branch_name'],
                    'branch2': line2_match['branch_name'],
                    'line1_street': line1_match['match_street'],
                    'line1_from': line1_match['match_from'],
                    'line1_to': line1_match['match_to'],
                    'line2_street': line2_match['match_street'],
                    'line2_from': line2_match['match_from'],
                    'line2_to': line2_match['match_to']
                }
                
                intersections_list.append(intersection_data)
                
                # Store intersection for line splitting
                if idx1 not in all_intersections:
                    all_intersections[idx1] = []
                all_intersections[idx1].append((x, y, z, seg_idx, param))
                
                # Also store for the second line
                # Find segment index and parameter for line 2
                seg_idx2, param2, _ = locate_point_on_line((x, y), shape2.points)
                if idx2 not in all_intersections:
                    all_intersections[idx2] = []
                all_intersections[idx2].append((x, y, z, seg_idx2, param2))
                
                # Create match description for console output
                line1_match_desc = []
                if line1_match['match_street']:
                    line1_match_desc.append("StreetName")
                if line1_match['match_from']:
                    line1_match_desc.append("From")
                if line1_match['match_to']:
                    line1_match_desc.append("To")
                
                line2_match_desc = []
                if line2_match['match_street']:
                    line2_match_desc.append("StreetName")
                if line2_match['match_from']:
                    line2_match_desc.append("From")
                if line2_match['match_to']:
                    line2_match_desc.append("To")
                
                line1_desc = ", ".join(line1_match_desc)
                line2_desc = ", ".join(line2_match_desc)
                
                print(f"Intersection found between '{line1_match['branch_name']}' ({line1_desc}) and '{line2_match['branch_name']}' ({line2_desc})")
    
    print(f"\nFound {len(intersections_list)} intersections between exactly matched lines")
    
    # Write intersections to shapefile
    for intersection in intersections_list:
        # Add point (with Z value if available)
        point = intersection['point']
        if len(point) > 2 and has_z:
            w_intersect.pointz(point[0], point[1], point[2])
        else:
            w_intersect.point(point[0], point[1])
        
        # Convert boolean values to Y/N for shapefile
        l1_street = "Y" if intersection['line1_street'] else "N"
        l1_from = "Y" if intersection['line1_from'] else "N"
        l1_to = "Y" if intersection['line1_to'] else "N"
        l2_street = "Y" if intersection['line2_street'] else "N"
        l2_from = "Y" if intersection['line2_from'] else "N"
        l2_to = "Y" if intersection['line2_to'] else "N"
        
        # Add record
        w_intersect.record(
            intersection['line1_idx'],
            intersection['line2_idx'],
            intersection['branch1'],
            intersection['branch2'],
            l1_street,
            l1_from,
            l1_to,
            l2_street,
            l2_from,
            l2_to
        )
    
    # Save the intersections shapefile
    w_intersect.close()
    
    # Copy the .prj file for intersections
    prj_path = shapefile_path.replace('.shp', '.prj')
    if os.path.exists(prj_path):
        try:
            output_prj = f"{intersections_output}.prj"
            shutil.copy2(prj_path, output_prj)
            print(f"Copied projection file to {output_prj}")
        except Exception as e:
            print(f"Warning: Could not copy projection file: {e}")
    
    print(f"Successfully created {intersections_output}.shp with {len(intersections_list)} intersection points")
    
    # Now create a new shapefile with the split lines
    print("\n--- Creating Split Line Shapefile ---")
    
    # Create a writer for the output shapefile, using the same shape type as original
    w_output = shapefile.Writer(output_path, shapeType=shape_type)
    
    # Copy field definitions from the original shapefile
    for field in fields:
        w_output.field(*field)
    
    # Add a field for the original feature ID
    w_output.field('ORIG_ID', 'N', 10, 0)
    # Add a field for the segment number (when a line is split into multiple segments)
    w_output.field('SEG_NUM', 'N', 10, 0)
    # Add a field for the branch name (if matched)
    w_output.field('BRANCH_NAME', 'C', 100, 0)
    
    # Process each shape
    # Process each shape
    for i, shape in enumerate(shapes):
        # Get the original record
        record = list(records[i])
        
        # Variables to track shapes actually written to the output
        segments_written = 0
        
        # Check if this shape has intersections
        if i in all_intersections and len(all_intersections[i]) > 0:
            # Get branch name if this is a matched line
            branch_name = ""
            if i in all_matches:
                branch_name = all_matches[i][0]['branch_name']
            
            # Get the points and Z values (if any)
            points = shape.points
            z_values = shape.z if has_z and hasattr(shape, 'z') else None
            
            # If we have Z values, combine XY with Z
            if has_z and z_values:
                points_3d = [(p[0], p[1], z) for p, z in zip(points, z_values)]
                
                # Split the line at intersections
                segments = split_line_at_intersections(points_3d, all_intersections[i], has_z=True)
                
                # Add each segment to the output shapefile
                for seg_num, segment in enumerate(segments):
                    if len(segment) >= 2:  # Only add segments with at least 2 points
                        try:
                            # Split into XY and Z
                            xy_points = [(p[0], p[1]) for p in segment]
                            z_vals = [p[2] for p in segment]
                            
                            # Combine XY and Z coordinates for linez method
                            points_with_z = []
                            for j, point in enumerate(xy_points):
                                x, y = point
                                z = z_vals[j] if j < len(z_vals) else 0
                                points_with_z.append((x, y, z))
                            
                            # Skip empty segments
                            if len(points_with_z) < 2:
                                print(f"Skipping invalid segment {i}-{seg_num} with {len(points_with_z)} points")
                                continue
                                
                            # Use the linez method with properly formatted 3D points
                            w_output.linez([points_with_z])
                            
                            # Add record for this segment - ONLY if the shape was written successfully
                            new_record = record + [i, seg_num, branch_name]
                            w_output.record(*new_record)
                            
                            segments_written += 1
                        except Exception as e:
                            print(f"Error writing segment {i}-{seg_num}: {e}")
                            # Don't increment segments_written since we failed
                
            else:
                # Split the line at intersections (no Z values)
                segments = split_line_at_intersections(points, all_intersections[i], has_z=False)
                
                # Add each segment to the output shapefile
                for seg_num, segment in enumerate(segments):
                    if len(segment) >= 2:  # Only add segments with at least 2 points
                        try:
                            # Write the line
                            w_output.line([segment])
                            
                            # Add record for this segment
                            new_record = record + [i, seg_num, branch_name]
                            w_output.record(*new_record)
                            
                            segments_written += 1
                        except Exception as e:
                            print(f"Error writing segment {i}-{seg_num}: {e}")
                            # Don't increment segments_written since we failed
            
            # If no valid segments were written for this shape, write a null shape
            if segments_written == 0:
                print(f"Warning: No valid segments for shape {i}, writing null shape to maintain sync")
                w_output.null()
                new_record = record + [i, 0, branch_name]
                w_output.record(*new_record)
                
        else:
            # No intersections, just copy the original shape and record
            branch_name = ""
            if i in all_matches:
                branch_name = all_matches[i][0]['branch_name']
            
            try:
                # Add the shape to the output
                # Handle different shape types appropriately
                if shape.shapeType == shapefile.POLYLINE:
                    if len(shape.points) >= 2:
                        w_output.line([shape.points])
                        # Add record with original ID
                        new_record = record + [i, 0, branch_name]
                        w_output.record(*new_record)
                    else:
                        print(f"Warning: Shape {i} has fewer than 2 points, writing null shape")
                        w_output.null()
                        new_record = record + [i, 0, branch_name]
                        w_output.record(*new_record)
                        
                elif shape.shapeType == shapefile.POLYLINEZ and hasattr(shape, 'z'):
                    if len(shape.points) >= 2:
                        # Combine XY and Z coordinates for linez method
                        points_with_z = []
                        for idx, point in enumerate(shape.points):
                            x, y = point
                            z = shape.z[idx] if idx < len(shape.z) else 0
                            points_with_z.append((x, y, z))
                        
                        # Use the linez method with properly formatted 3D points
                        w_output.linez([points_with_z])
                        
                        # Add record with original ID
                        new_record = record + [i, 0, branch_name]
                        w_output.record(*new_record)
                    else:
                        print(f"Warning: Shape {i} has fewer than 2 points, writing null shape")
                        w_output.null()
                        new_record = record + [i, 0, branch_name]
                        w_output.record(*new_record)
                else:
                    # Check if shape has enough points
                    if len(shape.points) >= 2:
                        # Default fallback
                        w_output.line([shape.points])
                        
                        # Add record with original ID
                        new_record = record + [i, 0, branch_name]
                        w_output.record(*new_record)
                    else:
                        print(f"Warning: Shape {i} has fewer than 2 points, writing null shape")
                        w_output.null()
                        new_record = record + [i, 0, branch_name]
                        w_output.record(*new_record)
            except Exception as e:
                print(f"Error writing shape {i}: {e}")
                # Write a null shape to maintain sync
                w_output.null()
                new_record = record + [i, 0, branch_name]
                w_output.record(*new_record)
    
    # Save the output shapefile
    w_output.close()
    
    # Copy the .prj file
    if os.path.exists(prj_path):
        try:
            output_prj = f"{output_path}.prj"
            shutil.copy2(prj_path, output_prj)
            print(f"Copied projection file to {output_prj}")
        except Exception as e:
            print(f"Warning: Could not copy projection file: {e}")
    
    print(f"Successfully created {output_path}.shp with split lines")

if __name__ == "__main__":
    # File paths
    lake_havasu_file = "Lake Havasu Sections - Jordan.xlsx"
    shapefile_data_file = "Shapefile Data.xlsx"
    shapefile_path = "Shapefile/LakeHavasuJordan.shp"
    output_folder = "Lake_Havasu_Analysis"
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Output file paths
    intersections_output = os.path.join(output_folder, "Intersections")
    split_lines_output = os.path.join(output_folder, "Split_Lines")
    
    # Run the analysis
    try:
        find_branch_name_segments(lake_havasu_file, shapefile_data_file, shapefile_path, 
                                   split_lines_output, intersections_output)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
