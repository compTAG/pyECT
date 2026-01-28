"""Tests for mesh processing functions in preprocessing/mesh_processing.py"""

import torch
import pytest
import tempfile
import os

# Import the function to test
try:
    from pyect import mesh_to_complex
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="trimesh not installed")
class TestMeshToComplex:
    """Tests for mesh_to_complex function."""

    def create_simple_obj_file(self, path):
        """Create a simple OBJ file with a triangle."""
        with open(path, 'w') as f:
            f.write("# Simple triangle\n")
            f.write("v 0.0 0.0 0.0\n")
            f.write("v 1.0 0.0 0.0\n")
            f.write("v 0.5 1.0 0.0\n")
            f.write("f 1 2 3\n")

    def create_tetrahedron_obj_file(self, path):
        """Create an OBJ file with a tetrahedron."""
        with open(path, 'w') as f:
            f.write("# Tetrahedron\n")
            f.write("v 0.0 0.0 0.0\n")
            f.write("v 1.0 0.0 0.0\n")
            f.write("v 0.5 1.0 0.0\n")
            f.write("v 0.5 0.5 1.0\n")
            f.write("f 1 2 3\n")
            f.write("f 1 2 4\n")
            f.write("f 1 3 4\n")
            f.write("f 2 3 4\n")

    def test_load_simple_triangle(self):
        """Test loading a simple triangle mesh."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write("v 0.0 0.0 0.0\n")
            f.write("v 1.0 0.0 0.0\n")
            f.write("v 0.5 1.0 0.0\n")
            f.write("f 1 2 3\n")
            temp_path = f.name

        try:
            c = mesh_to_complex(temp_path, torch.device("cpu"))

            # Should have 3 vertices
            assert c.get_coords(0).shape[0] == 3

            # Should be 3D coordinates
            assert c.get_coords(0).shape[1] == 3

            # Should have edges
            assert len(c) >= 2
        finally:
            os.unlink(temp_path)

    def test_output_is_complex(self):
        """Test that output is a Complex object."""
        from pyect import Complex

        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write("v 0.0 0.0 0.0\n")
            f.write("v 1.0 0.0 0.0\n")
            f.write("v 0.5 1.0 0.0\n")
            f.write("f 1 2 3\n")
            temp_path = f.name

        try:
            c = mesh_to_complex(temp_path, torch.device("cpu"))
            assert isinstance(c, Complex)
        finally:
            os.unlink(temp_path)

    def test_centering_option(self):
        """Test the centering option."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write("v 10.0 10.0 10.0\n")
            f.write("v 11.0 10.0 10.0\n")
            f.write("v 10.5 11.0 10.0\n")
            f.write("f 1 2 3\n")
            temp_path = f.name

        try:
            c = mesh_to_complex(temp_path, torch.device("cpu"), centering=True)

            v_coords = c.get_coords(0)
            centroid = v_coords.mean(dim=0)

            # Should be centered near origin
            assert torch.allclose(centroid, torch.zeros(3), atol=1e-5)
        finally:
            os.unlink(temp_path)

    def test_device_parameter(self):
        """Test the device parameter."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write("v 0.0 0.0 0.0\n")
            f.write("v 1.0 0.0 0.0\n")
            f.write("v 0.5 1.0 0.0\n")
            f.write("f 1 2 3\n")
            temp_path = f.name

        try:
            c = mesh_to_complex(temp_path, torch.device("cpu"))
            assert c.get_coords(0).device.type == "cpu"
        finally:
            os.unlink(temp_path)

    def test_vertex_weights_are_ones(self):
        """Test that default vertex weights are ones."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write("v 0.0 0.0 0.0\n")
            f.write("v 1.0 0.0 0.0\n")
            f.write("v 0.5 1.0 0.0\n")
            f.write("f 1 2 3\n")
            temp_path = f.name

        try:
            c = mesh_to_complex(temp_path, torch.device("cpu"))

            v_weights = c.get_weights(0)
            assert torch.allclose(v_weights, torch.ones(3))
        finally:
            os.unlink(temp_path)

    def test_tetrahedron_mesh(self):
        """Test loading a tetrahedron mesh."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write("v 0.0 0.0 0.0\n")
            f.write("v 1.0 0.0 0.0\n")
            f.write("v 0.5 1.0 0.0\n")
            f.write("v 0.5 0.5 1.0\n")
            f.write("f 1 2 3\n")
            f.write("f 1 2 4\n")
            f.write("f 1 3 4\n")
            f.write("f 2 3 4\n")
            temp_path = f.name

        try:
            c = mesh_to_complex(temp_path, torch.device("cpu"))

            # Should have 4 vertices
            assert c.get_coords(0).shape[0] == 4

            # Should have faces (dim 2)
            assert len(c) >= 3
        finally:
            os.unlink(temp_path)


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="trimesh not installed")
class TestMeshToComplexIntegration:
    """Integration tests for mesh_to_complex with WECT."""

    def test_mesh_with_wect(self):
        """Test that loaded mesh works with WECT."""
        from pyect import WECT

        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write("v 0.0 0.0 0.0\n")
            f.write("v 1.0 0.0 0.0\n")
            f.write("v 0.5 1.0 0.0\n")
            f.write("f 1 2 3\n")
            temp_path = f.name

        try:
            c = mesh_to_complex(temp_path, torch.device("cpu"))

            dirs = torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])
            wect = WECT(dirs, num_heights=10)

            result = wect(c)

            assert result.shape == (3, 10)
            assert torch.isfinite(result).all()
        finally:
            os.unlink(temp_path)
