import json
from io import BytesIO
import struct

import numpy as np
import pytest
import torch
from PIL import Image

from comfy.cli_args import args
import comfy_kitchen

args.cpu = True
if not hasattr(comfy_kitchen, "int8_attention_is_available"):
    comfy_kitchen.int8_attention_is_available = lambda: False

from comfy_api.latest import Types  # noqa: E402
from comfy_extras.nodes_save_3d import (  # noqa: E402
    MergeMeshes, MeshToFile3D, get_mesh_batch_item, mesh_item_to_glb_bytes, pack_variable_mesh_batch,
)


def _mesh(offset=0.0):
    vertices = torch.tensor([[[offset, 0.0, 0.0], [1.0 + offset, 0.0, 0.0], [offset, 1.0, 0.0]]])
    faces = torch.tensor([[[0, 1, 2]]], dtype=torch.long)
    uv = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    return Types.MESH(vertices, faces, uvs=uv, vertex_colors=torch.full((1, 3, 4), 0.25),
                      texture=torch.full((1, 2, 2, 3), 0.1), metallic_roughness=torch.full((1, 2, 2, 3), 0.2),
                      normals=torch.tensor([[[0.0, 0.0, 1.0]] * 3]), tangents=torch.tensor([[[1.0, 0.0, 0.0, 1.0]] * 3]),
                      normal_map=torch.full((1, 2, 2, 3), 0.5), occlusion_in_mr=True,
                      material={"double_sided": True, "metallic_factor": 0.7}, emissive=torch.full((1, 2, 2, 3), 0.3), unlit=True)


def test_mesh_to_file_rejects_multi_item_batch_instead_of_dropping_items():
    first, second = _mesh(), _mesh(2.0)
    batched = Types.MESH(torch.cat((first.vertices, second.vertices)), torch.cat((first.faces, second.faces)))
    with pytest.raises(ValueError, match="exactly one mesh item"):
        MeshToFile3D.execute(batched)


def test_mesh_to_file_rejects_empty_batch_explicitly():
    empty = Types.MESH(torch.empty((0, 3, 3)), torch.empty((0, 1, 3), dtype=torch.long))
    with pytest.raises(ValueError, match="exactly one mesh item"):
        MeshToFile3D.execute(empty)


@pytest.mark.parametrize("attribute", ["texture", "metallic_roughness", "normal_map", "emissive"])
def test_merge_meshes_rejects_mixed_image_attribute_presence(attribute):
    first, second = _mesh(), _mesh()
    setattr(second, attribute, None)
    with pytest.raises(ValueError, match=f"mixed presence of {attribute}"):
        MergeMeshes.execute({"mesh_0": first, "mesh_1": second})


def test_merge_meshes_preserves_all_mesh_attributes():
    first, second = _mesh(), _mesh(2.0)
    merged = MergeMeshes.execute({"mesh_0": first, "mesh_1": second}).result[0]
    assert merged.vertices.shape == (1, 6, 3)
    assert merged.faces.tolist() == [[[0, 1, 2], [3, 4, 5]]]
    assert merged.uvs.shape == (1, 6, 2) and merged.vertex_colors.shape == (1, 6, 4)
    assert merged.normals.shape == (1, 6, 3) and merged.tangents.shape == (1, 6, 4)
    assert torch.equal(merged.texture, first.texture) and torch.equal(merged.metallic_roughness, first.metallic_roughness)
    assert torch.equal(merged.normal_map, first.normal_map) and torch.equal(merged.emissive, first.emissive)
    assert merged.material == first.material and merged.occlusion_in_mr is True and merged.unlit is True


def test_variable_mesh_counts_select_the_requested_item():
    first, second = _mesh(), _mesh(2.0)
    packed = pack_variable_mesh_batch([first.vertices[0], second.vertices[0][:2]], [first.faces[0], second.faces[0][:0]],
                                      normals=[first.normals[0], second.normals[0][:2]])
    vertices, faces, _colors, _uvs, normals = get_mesh_batch_item(packed, 1)
    assert vertices.shape == (2, 3) and faces.shape == (0, 3) and normals.shape == (2, 3)
    assert packed.vertex_counts.tolist() == [3, 2] and packed.face_counts.tolist() == [1, 0]


def test_glb_round_trip_preserves_mesh_values_and_material_images():
    mesh = _mesh()
    mesh.unlit = False
    glb = mesh_item_to_glb_bytes(mesh, 0)
    json_length, _ = struct.unpack_from("<II", glb, 12)
    document = json.loads(glb[20:20 + json_length].decode("utf-8"))
    bin_header = 20 + json_length
    bin_length, _ = struct.unpack_from("<II", glb, bin_header)
    binary = glb[bin_header + 8:bin_header + 8 + bin_length]

    def accessor_values(name):
        primitive = document["meshes"][0]["primitives"][0]
        accessor = document["accessors"][primitive["attributes"][name]]
        view = document["bufferViews"][accessor["bufferView"]]
        width = {"VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        return np.frombuffer(binary, dtype=np.float32, count=accessor["count"] * width, offset=start).reshape(-1, width)

    primitive = document["meshes"][0]["primitives"][0]
    np.testing.assert_allclose(accessor_values("POSITION"), mesh.vertices[0].numpy())
    np.testing.assert_allclose(accessor_values("NORMAL"), mesh.normals[0].numpy())
    np.testing.assert_allclose(accessor_values("TANGENT"), mesh.tangents[0].numpy())
    np.testing.assert_allclose(accessor_values("TEXCOORD_0"), mesh.uvs[0].numpy())
    np.testing.assert_allclose(accessor_values("COLOR_0"), mesh.vertex_colors[0].numpy())
    assert accessor_values("POSITION").shape[0] == 3
    assert document["accessors"][primitive["indices"]]["count"] == 3
    material = document["materials"][0]
    assert primitive["material"] == 0 and material["doubleSided"] is True
    assert material["pbrMetallicRoughness"]["metallicFactor"] == pytest.approx(0.7)
    assert "baseColorTexture" in material["pbrMetallicRoughness"] and "metallicRoughnessTexture" in material["pbrMetallicRoughness"]
    assert "normalTexture" in material and "emissiveTexture" in material

    def image_pixels(texture_index):
        image = document["images"][document["textures"][texture_index]["source"]]
        view = document["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        return np.asarray(Image.open(BytesIO(binary[start:start + view["byteLength"]])))

    pbr = material["pbrMetallicRoughness"]
    np.testing.assert_array_equal(image_pixels(pbr["baseColorTexture"]["index"]), np.full((2, 2, 3), 25, dtype=np.uint8))
    np.testing.assert_array_equal(image_pixels(pbr["metallicRoughnessTexture"]["index"]), np.full((2, 2, 3), 51, dtype=np.uint8))
    np.testing.assert_array_equal(image_pixels(material["normalTexture"]["index"]), np.full((2, 2, 3), 127, dtype=np.uint8))
    np.testing.assert_array_equal(image_pixels(material["emissiveTexture"]["index"]), np.full((2, 2, 3), 76, dtype=np.uint8))
