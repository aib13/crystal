# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mesh.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)




DESCRIPTOR = _descriptor.FileDescriptor(
  name='mesh.proto',
  package='',
  serialized_pb='\n\nmesh.proto\"\x15\n\x05PData\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x05\"#\n\x0bPCoordinate\x12\t\n\x01x\x18\x01 \x02(\x01\x12\t\n\x01y\x18\x02 \x02(\x01\"\x17\n\x06PBound\x12\r\n\x05\x62ound\x18\x01 \x02(\x05\"H\n\x11PStructured2DGrid\x12\x10\n\x08num_rows\x18\x01 \x02(\x05\x12\x10\n\x08num_cols\x18\x02 \x02(\x05\x12\x0f\n\x07indices\x18\x03 \x03(\x05\"!\n\x0bPNeighbours\x12\x12\n\nelement_id\x18\x01 \x03(\x05\"\xb0\x01\n\x05PMesh\x12\x11\n\tnum_nodes\x18\x01 \x02(\x05\x12%\n\x0fnode2coordinate\x18\x02 \x03(\x0b\x32\x0c.PCoordinate\x12\x19\n\tnode2data\x18\x03 \x03(\x0b\x32\x06.PData\x12#\n\rnode2node_map\x18\x04 \x03(\x0b\x32\x0c.PNeighbours\x12-\n\x11structured_region\x18\x05 \x03(\x0b\x32\x12.PStructured2DGrid\"R\n\x15PStructuredNodeRegion\x12\x15\n\rregion_number\x18\x01 \x02(\x05\x12\x10\n\x08num_rows\x18\x02 \x02(\x05\x12\x10\n\x08num_cols\x18\x03 \x02(\x05\"\xa4\x01\n\x15PStructuredCellRegion\x12\x18\n\x10\x63\x65ll2node_offset\x18\x01 \x02(\x05\x12\x16\n\x0enode_row_start\x18\x02 \x02(\x05\x12\x17\n\x0fnode_row_finish\x18\x03 \x02(\x05\x12\x16\n\x0enode_col_start\x18\x04 \x02(\x05\x12\x17\n\x0fnode_col_finish\x18\x05 \x02(\x05\x12\x0f\n\x07\x63ompass\x18\x06 \x03(\x05\"\\\n\x17PUnstructuredCellRegion\x12\x1e\n\x16num_unstructured_cells\x18\x01 \x02(\x05\x12!\n\x19unstructured_cells_offset\x18\x02 \x02(\x05\"\xc1\x01\n\x15PStructuredEdgeRegion\x12\x1a\n\x12inedge2node_offset\x18\x01 \x02(\x05\x12\x16\n\x0enode_row_start\x18\x02 \x02(\x05\x12\x17\n\x0fnode_row_finish\x18\x03 \x02(\x05\x12\x16\n\x0enode_col_start\x18\x04 \x02(\x05\x12\x17\n\x0fnode_col_finish\x18\x05 \x02(\x05\x12\x14\n\x0cnode_compass\x18\x06 \x03(\x05\x12\x14\n\x0c\x63\x65ll_compass\x18\x07 \x03(\x05')




_PDATA = _descriptor.Descriptor(
  name='PData',
  full_name='PData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='PData.data', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=14,
  serialized_end=35,
)


_PCOORDINATE = _descriptor.Descriptor(
  name='PCoordinate',
  full_name='PCoordinate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='PCoordinate.x', index=0,
      number=1, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='y', full_name='PCoordinate.y', index=1,
      number=2, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=37,
  serialized_end=72,
)


_PBOUND = _descriptor.Descriptor(
  name='PBound',
  full_name='PBound',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='bound', full_name='PBound.bound', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=74,
  serialized_end=97,
)


_PSTRUCTURED2DGRID = _descriptor.Descriptor(
  name='PStructured2DGrid',
  full_name='PStructured2DGrid',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_rows', full_name='PStructured2DGrid.num_rows', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_cols', full_name='PStructured2DGrid.num_cols', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='indices', full_name='PStructured2DGrid.indices', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=99,
  serialized_end=171,
)


_PNEIGHBOURS = _descriptor.Descriptor(
  name='PNeighbours',
  full_name='PNeighbours',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='element_id', full_name='PNeighbours.element_id', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=173,
  serialized_end=206,
)


_PMESH = _descriptor.Descriptor(
  name='PMesh',
  full_name='PMesh',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_nodes', full_name='PMesh.num_nodes', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node2coordinate', full_name='PMesh.node2coordinate', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node2data', full_name='PMesh.node2data', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node2node_map', full_name='PMesh.node2node_map', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='structured_region', full_name='PMesh.structured_region', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=209,
  serialized_end=385,
)


_PSTRUCTUREDNODEREGION = _descriptor.Descriptor(
  name='PStructuredNodeRegion',
  full_name='PStructuredNodeRegion',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='region_number', full_name='PStructuredNodeRegion.region_number', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_rows', full_name='PStructuredNodeRegion.num_rows', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_cols', full_name='PStructuredNodeRegion.num_cols', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=387,
  serialized_end=469,
)


_PSTRUCTUREDCELLREGION = _descriptor.Descriptor(
  name='PStructuredCellRegion',
  full_name='PStructuredCellRegion',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='cell2node_offset', full_name='PStructuredCellRegion.cell2node_offset', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node_row_start', full_name='PStructuredCellRegion.node_row_start', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node_row_finish', full_name='PStructuredCellRegion.node_row_finish', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node_col_start', full_name='PStructuredCellRegion.node_col_start', index=3,
      number=4, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node_col_finish', full_name='PStructuredCellRegion.node_col_finish', index=4,
      number=5, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='compass', full_name='PStructuredCellRegion.compass', index=5,
      number=6, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=472,
  serialized_end=636,
)


_PUNSTRUCTUREDCELLREGION = _descriptor.Descriptor(
  name='PUnstructuredCellRegion',
  full_name='PUnstructuredCellRegion',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_unstructured_cells', full_name='PUnstructuredCellRegion.num_unstructured_cells', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='unstructured_cells_offset', full_name='PUnstructuredCellRegion.unstructured_cells_offset', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=638,
  serialized_end=730,
)


_PSTRUCTUREDEDGEREGION = _descriptor.Descriptor(
  name='PStructuredEdgeRegion',
  full_name='PStructuredEdgeRegion',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='inedge2node_offset', full_name='PStructuredEdgeRegion.inedge2node_offset', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node_row_start', full_name='PStructuredEdgeRegion.node_row_start', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node_row_finish', full_name='PStructuredEdgeRegion.node_row_finish', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node_col_start', full_name='PStructuredEdgeRegion.node_col_start', index=3,
      number=4, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node_col_finish', full_name='PStructuredEdgeRegion.node_col_finish', index=4,
      number=5, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node_compass', full_name='PStructuredEdgeRegion.node_compass', index=5,
      number=6, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cell_compass', full_name='PStructuredEdgeRegion.cell_compass', index=6,
      number=7, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=733,
  serialized_end=926,
)

_PMESH.fields_by_name['node2coordinate'].message_type = _PCOORDINATE
_PMESH.fields_by_name['node2data'].message_type = _PDATA
_PMESH.fields_by_name['node2node_map'].message_type = _PNEIGHBOURS
_PMESH.fields_by_name['structured_region'].message_type = _PSTRUCTURED2DGRID
DESCRIPTOR.message_types_by_name['PData'] = _PDATA
DESCRIPTOR.message_types_by_name['PCoordinate'] = _PCOORDINATE
DESCRIPTOR.message_types_by_name['PBound'] = _PBOUND
DESCRIPTOR.message_types_by_name['PStructured2DGrid'] = _PSTRUCTURED2DGRID
DESCRIPTOR.message_types_by_name['PNeighbours'] = _PNEIGHBOURS
DESCRIPTOR.message_types_by_name['PMesh'] = _PMESH
DESCRIPTOR.message_types_by_name['PStructuredNodeRegion'] = _PSTRUCTUREDNODEREGION
DESCRIPTOR.message_types_by_name['PStructuredCellRegion'] = _PSTRUCTUREDCELLREGION
DESCRIPTOR.message_types_by_name['PUnstructuredCellRegion'] = _PUNSTRUCTUREDCELLREGION
DESCRIPTOR.message_types_by_name['PStructuredEdgeRegion'] = _PSTRUCTUREDEDGEREGION

class PData(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _PDATA

  # @@protoc_insertion_point(class_scope:PData)

class PCoordinate(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _PCOORDINATE

  # @@protoc_insertion_point(class_scope:PCoordinate)

class PBound(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _PBOUND

  # @@protoc_insertion_point(class_scope:PBound)

class PStructured2DGrid(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _PSTRUCTURED2DGRID

  # @@protoc_insertion_point(class_scope:PStructured2DGrid)

class PNeighbours(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _PNEIGHBOURS

  # @@protoc_insertion_point(class_scope:PNeighbours)

class PMesh(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _PMESH

  # @@protoc_insertion_point(class_scope:PMesh)

class PStructuredNodeRegion(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _PSTRUCTUREDNODEREGION

  # @@protoc_insertion_point(class_scope:PStructuredNodeRegion)

class PStructuredCellRegion(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _PSTRUCTUREDCELLREGION

  # @@protoc_insertion_point(class_scope:PStructuredCellRegion)

class PUnstructuredCellRegion(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _PUNSTRUCTUREDCELLREGION

  # @@protoc_insertion_point(class_scope:PUnstructuredCellRegion)

class PStructuredEdgeRegion(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _PSTRUCTUREDEDGEREGION

  # @@protoc_insertion_point(class_scope:PStructuredEdgeRegion)


# @@protoc_insertion_point(module_scope)