# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: challenge.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x63hallenge.proto\x12\x12\x63hallenge_protocol\"u\n\x17SimulationHistoryBuffer\x12\x12\n\nego_states\x18\x01 \x03(\x0c\x12\x14\n\x0cobservations\x18\x02 \x03(\x0c\x12\x1c\n\x0fsample_interval\x18\x03 \x01(\x02H\x00\x88\x01\x01\x42\x12\n\x10_sample_interval\"5\n\x13SimulationIteration\x12\x0f\n\x07time_us\x18\x01 \x01(\x03\x12\r\n\x05index\x18\x02 \x01(\x05\"\xa5\x01\n\x0cPlannerInput\x12\x45\n\x14simulation_iteration\x18\x01 \x01(\x0b\x32\'.challenge_protocol.SimulationIteration\x12N\n\x19simulation_history_buffer\x18\x02 \x01(\x0b\x32+.challenge_protocol.SimulationHistoryBuffer\"M\n\x11MultiPlannerInput\x12\x38\n\x0eplanner_inputs\x18\x01 \x03(\x0b\x32 .challenge_protocol.PlannerInput\"1\n\x08StateSE2\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\x0f\n\x07heading\x18\x03 \x01(\x02\"%\n\rStateVector2D\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\"\xb8\x01\n\x1aPlannerInitializationLight\x12\x37\n\x11\x65xpert_goal_state\x18\x01 \x01(\x0b\x32\x1c.challenge_protocol.StateSE2\x12\x1b\n\x13route_roadblock_ids\x18\x02 \x03(\t\x12\x32\n\x0cmission_goal\x18\x03 \x01(\x0b\x32\x1c.challenge_protocol.StateSE2\x12\x10\n\x08map_name\x18\x04 \x01(\t\"r\n\x1fMultiPlannerInitializationLight\x12O\n\x17planner_initializations\x18\x01 \x03(\x0b\x32..challenge_protocol.PlannerInitializationLight\"?\n\x1dPlannerInitializationResponse\x12\x1e\n\x16\x63onsume_batched_inputs\x18\x01 \x01(\x08\"\xa2\x02\n\x08\x45goState\x12\x34\n\x0erear_axle_pose\x18\x01 \x01(\x0b\x32\x1c.challenge_protocol.StateSE2\x12@\n\x15rear_axle_velocity_2d\x18\x02 \x01(\x0b\x32!.challenge_protocol.StateVector2D\x12\x44\n\x19rear_axle_acceleration_2d\x18\x03 \x01(\x0b\x32!.challenge_protocol.StateVector2D\x12\x1b\n\x13tire_steering_angle\x18\x04 \x01(\x02\x12\x0f\n\x07time_us\x18\x05 \x01(\x03\x12\x13\n\x0b\x61ngular_vel\x18\x06 \x01(\x02\x12\x15\n\rangular_accel\x18\x07 \x01(\x02\">\n\nTrajectory\x12\x30\n\nego_states\x18\x01 \x03(\x0b\x32\x1c.challenge_protocol.EgoState\"G\n\x0fMultiTrajectory\x12\x34\n\x0ctrajectories\x18\x01 \x03(\x0b\x32\x1e.challenge_protocol.Trajectory2\xfc\x01\n\x18\x44\x65tectionTracksChallenge\x12}\n\x11InitializePlanner\x12\x33.challenge_protocol.MultiPlannerInitializationLight\x1a\x31.challenge_protocol.PlannerInitializationResponse\"\x00\x12\x61\n\x11\x43omputeTrajectory\x12%.challenge_protocol.MultiPlannerInput\x1a#.challenge_protocol.MultiTrajectory\"\x00\x62\x06proto3')



_SIMULATIONHISTORYBUFFER = DESCRIPTOR.message_types_by_name['SimulationHistoryBuffer']
_SIMULATIONITERATION = DESCRIPTOR.message_types_by_name['SimulationIteration']
_PLANNERINPUT = DESCRIPTOR.message_types_by_name['PlannerInput']
_MULTIPLANNERINPUT = DESCRIPTOR.message_types_by_name['MultiPlannerInput']
_STATESE2 = DESCRIPTOR.message_types_by_name['StateSE2']
_STATEVECTOR2D = DESCRIPTOR.message_types_by_name['StateVector2D']
_PLANNERINITIALIZATIONLIGHT = DESCRIPTOR.message_types_by_name['PlannerInitializationLight']
_MULTIPLANNERINITIALIZATIONLIGHT = DESCRIPTOR.message_types_by_name['MultiPlannerInitializationLight']
_PLANNERINITIALIZATIONRESPONSE = DESCRIPTOR.message_types_by_name['PlannerInitializationResponse']
_EGOSTATE = DESCRIPTOR.message_types_by_name['EgoState']
_TRAJECTORY = DESCRIPTOR.message_types_by_name['Trajectory']
_MULTITRAJECTORY = DESCRIPTOR.message_types_by_name['MultiTrajectory']
SimulationHistoryBuffer = _reflection.GeneratedProtocolMessageType('SimulationHistoryBuffer', (_message.Message,), {
  'DESCRIPTOR' : _SIMULATIONHISTORYBUFFER,
  '__module__' : 'challenge_pb2'
  # @@protoc_insertion_point(class_scope:challenge_protocol.SimulationHistoryBuffer)
  })
_sym_db.RegisterMessage(SimulationHistoryBuffer)

SimulationIteration = _reflection.GeneratedProtocolMessageType('SimulationIteration', (_message.Message,), {
  'DESCRIPTOR' : _SIMULATIONITERATION,
  '__module__' : 'challenge_pb2'
  # @@protoc_insertion_point(class_scope:challenge_protocol.SimulationIteration)
  })
_sym_db.RegisterMessage(SimulationIteration)

PlannerInput = _reflection.GeneratedProtocolMessageType('PlannerInput', (_message.Message,), {
  'DESCRIPTOR' : _PLANNERINPUT,
  '__module__' : 'challenge_pb2'
  # @@protoc_insertion_point(class_scope:challenge_protocol.PlannerInput)
  })
_sym_db.RegisterMessage(PlannerInput)

MultiPlannerInput = _reflection.GeneratedProtocolMessageType('MultiPlannerInput', (_message.Message,), {
  'DESCRIPTOR' : _MULTIPLANNERINPUT,
  '__module__' : 'challenge_pb2'
  # @@protoc_insertion_point(class_scope:challenge_protocol.MultiPlannerInput)
  })
_sym_db.RegisterMessage(MultiPlannerInput)

StateSE2 = _reflection.GeneratedProtocolMessageType('StateSE2', (_message.Message,), {
  'DESCRIPTOR' : _STATESE2,
  '__module__' : 'challenge_pb2'
  # @@protoc_insertion_point(class_scope:challenge_protocol.StateSE2)
  })
_sym_db.RegisterMessage(StateSE2)

StateVector2D = _reflection.GeneratedProtocolMessageType('StateVector2D', (_message.Message,), {
  'DESCRIPTOR' : _STATEVECTOR2D,
  '__module__' : 'challenge_pb2'
  # @@protoc_insertion_point(class_scope:challenge_protocol.StateVector2D)
  })
_sym_db.RegisterMessage(StateVector2D)

PlannerInitializationLight = _reflection.GeneratedProtocolMessageType('PlannerInitializationLight', (_message.Message,), {
  'DESCRIPTOR' : _PLANNERINITIALIZATIONLIGHT,
  '__module__' : 'challenge_pb2'
  # @@protoc_insertion_point(class_scope:challenge_protocol.PlannerInitializationLight)
  })
_sym_db.RegisterMessage(PlannerInitializationLight)

MultiPlannerInitializationLight = _reflection.GeneratedProtocolMessageType('MultiPlannerInitializationLight', (_message.Message,), {
  'DESCRIPTOR' : _MULTIPLANNERINITIALIZATIONLIGHT,
  '__module__' : 'challenge_pb2'
  # @@protoc_insertion_point(class_scope:challenge_protocol.MultiPlannerInitializationLight)
  })
_sym_db.RegisterMessage(MultiPlannerInitializationLight)

PlannerInitializationResponse = _reflection.GeneratedProtocolMessageType('PlannerInitializationResponse', (_message.Message,), {
  'DESCRIPTOR' : _PLANNERINITIALIZATIONRESPONSE,
  '__module__' : 'challenge_pb2'
  # @@protoc_insertion_point(class_scope:challenge_protocol.PlannerInitializationResponse)
  })
_sym_db.RegisterMessage(PlannerInitializationResponse)

EgoState = _reflection.GeneratedProtocolMessageType('EgoState', (_message.Message,), {
  'DESCRIPTOR' : _EGOSTATE,
  '__module__' : 'challenge_pb2'
  # @@protoc_insertion_point(class_scope:challenge_protocol.EgoState)
  })
_sym_db.RegisterMessage(EgoState)

Trajectory = _reflection.GeneratedProtocolMessageType('Trajectory', (_message.Message,), {
  'DESCRIPTOR' : _TRAJECTORY,
  '__module__' : 'challenge_pb2'
  # @@protoc_insertion_point(class_scope:challenge_protocol.Trajectory)
  })
_sym_db.RegisterMessage(Trajectory)

MultiTrajectory = _reflection.GeneratedProtocolMessageType('MultiTrajectory', (_message.Message,), {
  'DESCRIPTOR' : _MULTITRAJECTORY,
  '__module__' : 'challenge_pb2'
  # @@protoc_insertion_point(class_scope:challenge_protocol.MultiTrajectory)
  })
_sym_db.RegisterMessage(MultiTrajectory)

_DETECTIONTRACKSCHALLENGE = DESCRIPTOR.services_by_name['DetectionTracksChallenge']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _SIMULATIONHISTORYBUFFER._serialized_start=39
  _SIMULATIONHISTORYBUFFER._serialized_end=156
  _SIMULATIONITERATION._serialized_start=158
  _SIMULATIONITERATION._serialized_end=211
  _PLANNERINPUT._serialized_start=214
  _PLANNERINPUT._serialized_end=379
  _MULTIPLANNERINPUT._serialized_start=381
  _MULTIPLANNERINPUT._serialized_end=458
  _STATESE2._serialized_start=460
  _STATESE2._serialized_end=509
  _STATEVECTOR2D._serialized_start=511
  _STATEVECTOR2D._serialized_end=548
  _PLANNERINITIALIZATIONLIGHT._serialized_start=551
  _PLANNERINITIALIZATIONLIGHT._serialized_end=735
  _MULTIPLANNERINITIALIZATIONLIGHT._serialized_start=737
  _MULTIPLANNERINITIALIZATIONLIGHT._serialized_end=851
  _PLANNERINITIALIZATIONRESPONSE._serialized_start=853
  _PLANNERINITIALIZATIONRESPONSE._serialized_end=916
  _EGOSTATE._serialized_start=919
  _EGOSTATE._serialized_end=1209
  _TRAJECTORY._serialized_start=1211
  _TRAJECTORY._serialized_end=1273
  _MULTITRAJECTORY._serialized_start=1275
  _MULTITRAJECTORY._serialized_end=1346
  _DETECTIONTRACKSCHALLENGE._serialized_start=1349
  _DETECTIONTRACKSCHALLENGE._serialized_end=1601
# @@protoc_insertion_point(module_scope)
