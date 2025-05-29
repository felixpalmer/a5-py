import pytest
import numpy as np
from a5.core.cell import lonlat_to_cell, cell_to_lonlat, cell_to_boundary, a5cell_contains_point
from a5.core.coordinate_systems import LonLat
from a5.core.utils import A5Cell
from a5.core.serialization import deserialize, MAX_RESOLUTION

def test_cell_boundary_contains_point():
    # Test coordinates for Ho Chi Minh City
    hcmc_lonlat: LonLat = (106.706360, 10.775305)
    
    # Dictionary to store failures for each resolution
    failures = {}
    
    # Test resolutions from 0 to MAX_RESOLUTION
    for resolution in range(MAX_RESOLUTION + 1):
        resolution_failures = []
        
        try:
            # Get cell ID for the coordinates
            cell_id = lonlat_to_cell(hcmc_lonlat, resolution)
            
            # Get cell boundary
            boundary = cell_to_boundary(cell_id)
            
            # Verify the original point is contained within the cell
            cell = deserialize(cell_id)
            if not a5cell_contains_point(cell, hcmc_lonlat):
                resolution_failures.append(f"Cell does not contain the original point {hcmc_lonlat}")
                # Add cell center for reference
                center = cell_to_lonlat(cell_id)
                resolution_failures.append(f"Cell center is at {center}")
            
            # Verify boundary points are valid coordinates
            for i, point in enumerate(boundary):
                if not isinstance(point, tuple):
                    resolution_failures.append(f"Boundary point {i} is not a tuple, got {type(point)}")
                elif len(point) != 2:
                    resolution_failures.append(f"Boundary point {i} should have 2 coordinates, got {len(point)}")
                else:
                    lon, lat = point
                    if not (-180 <= lon <= 180):
                        resolution_failures.append(f"Boundary point {i} has invalid longitude: {lon}")
                    if not (-90 <= lat <= 90):
                        resolution_failures.append(f"Boundary point {i} has invalid latitude: {lat}")
            
            # Add boundary information for debugging
            if resolution_failures:
                resolution_failures.append(f"Boundary points: {boundary}")
                        
        except Exception as e:
            resolution_failures.append(f"Unexpected error: {str(e)}")
            if hasattr(e, '__traceback__'):
                import traceback
                resolution_failures.append(f"Traceback: {''.join(traceback.format_tb(e.__traceback__))}")
            
        # Store failures for this resolution if any occurred
        if resolution_failures:
            failures[resolution] = resolution_failures
    
    # Report all failures
    if failures:
        failure_message = "\nFailures by resolution:\n"
        for resolution, resolution_failures in failures.items():
            failure_message += f"\nResolution {resolution}:\n"
            for failure in resolution_failures:
                failure_message += f"  - {failure}\n"
        pytest.fail(failure_message)
    else:
        assert True, "All resolutions passed all checks"

if __name__ == "__main__":
    test_cell_boundary_contains_point() 