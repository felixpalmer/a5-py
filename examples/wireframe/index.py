#!/usr/bin/env python3

import sys
import json
import random
import os

# Add the a5-py directory to the path so we can import the a5 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from a5.core.cell import cell_to_boundary
from a5.core.hex import u64_to_hex
from a5.core.serialization import cell_to_children, WORLD_CELL


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: python index.py <resolution> <output.json>\n")
        sys.stderr.write("  resolution: A5 cell resolution (integer)\n")
        sys.exit(1)
    
    try:
        resolution = int(sys.argv[1])
        output_file = sys.argv[2]
    except ValueError:
        sys.stderr.write("Error: resolution must be an integer\n")
        sys.exit(1)
    
    cells = []
    try:
        # Calculate total number of cells at this resolution
        cell_ids = cell_to_children(WORLD_CELL, resolution)
        
        # Generate all cells
        for cell_id in cell_ids:
            boundary = cell_to_boundary(cell_id, {
                'closed_ring': True,
                'segments': 1,
            })
            
            cells.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [boundary],
                },
                "properties": {
                    "cellIdHex": u64_to_hex(cell_id),
                },
            })
        
        # Create GeoJSON FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "features": cells,
        }
        
        # Write to JSON file
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print("Successfully generated {} A5 cells at resolution {}".format(len(cells), resolution))
        print("Output written to {}".format(output_file))
        
    except Exception as error:
        sys.stderr.write("Error generating cells: {}\n".format(error))
        sys.exit(1)


if __name__ == "__main__":
    main() 
