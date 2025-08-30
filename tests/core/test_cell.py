# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
import pytest
from pathlib import Path
from typing import List, Tuple, Dict, Any

from a5.core.coordinate_systems import Degrees, LonLat
from a5.core.cell import cell_to_boundary, cell_to_lonlat, lonlat_to_cell, a5cell_contains_point
from a5.core.serialization import deserialize, MAX_RESOLUTION
from a5.core.hex import hex_to_u64, u64_to_hex

# Load test data
FIXTURES_PATH = Path(__file__).parent / "fixtures"
POPULATED_PLACES_PATH = FIXTURES_PATH / "ne_50m_populated_places_nameonly.json"

with open(POPULATED_PLACES_PATH) as f:
    populated_places = json.load(f)
    populated_places['features'] = populated_places['features']

class TestAntimeridianCells:
    """Test antimeridian crossing behavior."""
    
    def test_antimeridian_cell_longitude_span(self):
        """Antimeridian cell should have longitude span less than 180 degrees."""
        antimeridian_cells = ['eb60000000000000', '2e00000000000000']
        segments = [1, 10, 'auto']
        
        for cell_id in antimeridian_cells:
            for segment in segments:
                cell_id_bigint = hex_to_u64(cell_id)
                boundary = cell_to_boundary(cell_id_bigint, {'segments': segment if segment != 'auto' else None})

                # Check for antimeridian crossing
                longitudes = [lon for lon, lat in boundary]
                min_lon = min(longitudes)
                max_lon = max(longitudes)
                lon_span = max_lon - min_lon
                assert lon_span < 180, f"Cell {cell_id} with {segment} segments has longitude span {lon_span} >= 180"

class TestCellBoundary:
    """Test cell boundary containment functionality."""
    
    def test_cell_contains_original_point_for_all_resolutions(self):
        """Test that cells contain their original points for all resolutions."""
        # Extract coordinates from GeoJSON features
        test_points: List[LonLat] = []
        for feature in populated_places['features']:
            lon, lat = feature['geometry']['coordinates']
            test_points.append((lon, lat))

        print(f"Testing with {len(test_points)} points from GeoJSON file")

        # Dictionary to store failures for each resolution and point
        failures: Dict[str, Dict[int, List[str]]] = {}

        print(f"Skipping resolution {MAX_RESOLUTION} as lonlat_to_cell is not implemented for this resolution yet")
        
        # Test each point from GeoJSON
        for point_index, test_lonlat in enumerate(test_points):
            feature_name = populated_places['features'][point_index]['properties'].get('name', f'Unnamed {point_index}')
            point_key = f"Point {point_index} - {feature_name} ({test_lonlat[0]}, {test_lonlat[1]})"

            # Test resolutions from 1 to MAX_RESOLUTION - 1
            for resolution in range(1, MAX_RESOLUTION):
                if resolution == MAX_RESOLUTION or abs(test_lonlat[1]) > 80: # Issues in polar regions, TODO fix
                    continue

                resolution_failures: List[str] = []
                
                try:
                    # Get cell ID for the coordinates
                    cell_id = lonlat_to_cell(test_lonlat, resolution)
                    
                    # Verify the original point is contained within the cell
                    cell = deserialize(cell_id)
                    if (a5cell_contains_point(cell, test_lonlat) < 0):
                        # Get cell boundary
                        boundary = cell_to_boundary(cell_id)
                        
                        # Convert boundary to GeoJSON
                        geojson = self._boundary_to_geojson(boundary, resolution, u64_to_hex(cell_id), test_lonlat)
                        
                        resolution_failures.append(f"Cell {cell_id} does not contain the original point {test_lonlat}")
                        resolution_failures.append(f"GeoJSON:\n {json.dumps(geojson)}")
                    
                except Exception as e:
                    resolution_failures.append(f"Unexpected error: {e}")
                    import traceback
                    resolution_failures.append(f"Traceback: {traceback.format_exc()}")
                
                # Store failures for this resolution if any occurred
                if resolution_failures:
                    if point_key not in failures:
                        failures[point_key] = {}
                    failures[point_key][resolution] = resolution_failures

        # Report all failures
        if failures:
            failure_message = '\nFailures by point and resolution:\n'
            for point_key, point_failures in failures.items():
                if point_failures:
                    failure_message += f'\n{point_key}:\n'
                    for resolution, resolution_failures in point_failures.items():
                        failure_message += f'  Resolution {resolution}:\n'
                        for failure in resolution_failures:
                            failure_message += f'    - {failure}\n'
            raise AssertionError(failure_message)

    def _boundary_to_geojson(self, boundary: List[LonLat], resolution: int, cell_id: str, origin_point: LonLat) -> Dict[str, Any]:
        """Convert boundary to GeoJSON format."""
        # Create coordinates list with first point appended at the end to close the polygon
        coordinates = [[lon, lat] for lon, lat in boundary]

        # Create a polygon feature for the cell
        cell_feature = {
            'type': 'Feature',
            'properties': {
                'resolution': resolution,
                'cell_id': cell_id,
                'origin_point': f"{origin_point[0]},{origin_point[1]}"
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [coordinates]  # Wrap in list as per GeoJSON spec
            }
        }

        # Create a point feature for the origin point
        point_feature = {
            'type': 'Feature',
            'properties': {
                'resolution': resolution,
                'cell_id': cell_id,
                'origin_point': f"{origin_point[0]},{origin_point[1]}"
            },
            'geometry': {
                'type': 'Point',
                'coordinates': list(origin_point)
            }
        }

        # Create a feature collection with both features
        feature_collection = {
            'type': 'FeatureCollection',
            'features': [cell_feature, point_feature]
        }

        return feature_collection 