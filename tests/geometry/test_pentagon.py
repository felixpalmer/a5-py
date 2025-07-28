import pytest
import json
from pathlib import Path

from a5.geometry.pentagon import PentagonShape


def load_fixtures():
    """Load pentagon test fixtures."""
    fixture_path = Path(__file__).parent / "fixtures" / "pentagon.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


class TestPentagonShape:
    """Test cases for PentagonShape class."""
    
    def test_contains_point(self):
        """Test that containsPoint returns correct results for all test cases."""
        fixtures = load_fixtures()
        
        for fixture in fixtures:
            pentagon = PentagonShape([tuple(vertex) for vertex in fixture["vertices"]])
            
            for test_case in fixture["containsPointTests"]:
                point = tuple(test_case["point"])
                expected = test_case["result"]
                actual = pentagon.contains_point(point)
                assert abs(actual - expected) < 1e-6, f"Contains point test failed: expected {expected}, got {actual}"

    def test_get_area(self):
        """Test that getArea returns correct area for all pentagons."""
        fixtures = load_fixtures()
        
        for fixture in fixtures:
            pentagon = PentagonShape([tuple(vertex) for vertex in fixture["vertices"]])
            area = pentagon.get_area()
            expected = fixture["area"]
            assert abs(area - expected) < 1e-6, f"Area test failed: expected {expected}, got {area}"

    def test_get_center(self):
        """Test that getCenter returns correct center for all pentagons."""
        fixtures = load_fixtures()
        
        for fixture in fixtures:
            pentagon = PentagonShape([tuple(vertex) for vertex in fixture["vertices"]])
            center = pentagon.get_center()
            expected = fixture["center"]
            assert abs(center[0] - expected[0]) < 1e-6, f"Center X test failed: expected {expected[0]}, got {center[0]}"
            assert abs(center[1] - expected[1]) < 1e-6, f"Center Y test failed: expected {expected[1]}, got {center[1]}"

    def test_transformations(self):
        """Test that all transformations work correctly."""
        fixtures = load_fixtures()
        
        for fixture in fixtures:
            # Test scale transformation
            pentagon = PentagonShape([tuple(vertex) for vertex in fixture["vertices"]])
            scaled = pentagon.clone().scale(2)
            vertices = scaled.get_vertices()
            
            expected_scale = fixture["transformTests"]["scale"]
            for i, expected in enumerate(expected_scale):
                assert abs(vertices[i][0] - expected[0]) < 1e-6, f"Scale X test failed at vertex {i}"
                assert abs(vertices[i][1] - expected[1]) < 1e-6, f"Scale Y test failed at vertex {i}"

            # Test rotate180 transformation
            pentagon = PentagonShape([tuple(vertex) for vertex in fixture["vertices"]])
            rotated = pentagon.clone().rotate180()
            vertices = rotated.get_vertices()
            
            expected_rotate = fixture["transformTests"]["rotate180"]
            for i, expected in enumerate(expected_rotate):
                assert abs(vertices[i][0] - expected[0]) < 1e-6, f"Rotate180 X test failed at vertex {i}"
                assert abs(vertices[i][1] - expected[1]) < 1e-6, f"Rotate180 Y test failed at vertex {i}"

            # Test reflectY transformation
            pentagon = PentagonShape([tuple(vertex) for vertex in fixture["vertices"]])
            reflected = pentagon.clone().reflect_y()
            vertices = reflected.get_vertices()
            
            expected_reflect = fixture["transformTests"]["reflectY"]
            for i, expected in enumerate(expected_reflect):
                assert abs(vertices[i][0] - expected[0]) < 1e-6, f"ReflectY X test failed at vertex {i}"
                assert abs(vertices[i][1] - expected[1]) < 1e-6, f"ReflectY Y test failed at vertex {i}"

            # Test translate transformation
            pentagon = PentagonShape([tuple(vertex) for vertex in fixture["vertices"]])
            translated = pentagon.clone().translate((1, 1))
            vertices = translated.get_vertices()
            
            expected_translate = fixture["transformTests"]["translate"]
            for i, expected in enumerate(expected_translate):
                assert abs(vertices[i][0] - expected[0]) < 1e-6, f"Translate X test failed at vertex {i}"
                assert abs(vertices[i][1] - expected[1]) < 1e-6, f"Translate Y test failed at vertex {i}"

    def test_split_edges(self):
        """Test that splitEdges returns split edges with different segment counts."""
        fixtures = load_fixtures()
        
        for fixture in fixtures:
            pentagon = PentagonShape([tuple(vertex) for vertex in fixture["vertices"]])
            
            # Test boundaries with 2-3 segments
            for n_segments in [2, 3]:
                split = pentagon.clone().split_edges(n_segments)
                vertices = split.get_vertices()
                expected_vertices = fixture["splitEdgesTests"][f"segments{n_segments}"]
                
                assert len(vertices) == len(expected_vertices), f"Split edges vertex count mismatch for {n_segments} segments"
                
                for i, expected in enumerate(expected_vertices):
                    assert abs(vertices[i][0] - expected[0]) < 1e-6, f"Split edges X test failed at vertex {i} for {n_segments} segments"
                    assert abs(vertices[i][1] - expected[1]) < 1e-6, f"Split edges Y test failed at vertex {i} for {n_segments} segments" 