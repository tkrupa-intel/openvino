// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This collection contains one entry for each op. If an op is added it must be
// added to this list.
//
// In order to use this list you want to define a macro named exactly _OPENVINO_OP_REG
// When you are done you should undef the macro
// As an example if you wanted to make a list of all op names as strings you could do this:

#ifndef _OPENVINO_OP_REG
#    warning "_OPENVINO_OP_REG not defined"
#    define _OPENVINO_OP_REG(x, y)
#endif

_OPENVINO_OP_REG(Abs, ngraph::op::v0)
_OPENVINO_OP_REG(Acos, ngraph::op::v0)
_OPENVINO_OP_REG(Acosh, ngraph::op::v3)
_OPENVINO_OP_REG(Add, ngraph::op::v1)
_OPENVINO_OP_REG(Asin, ngraph::op::v0)
_OPENVINO_OP_REG(Asinh, ngraph::op::v3)
_OPENVINO_OP_REG(Atan, ngraph::op::v0)
_OPENVINO_OP_REG(Atanh, ngraph::op::v3)
_OPENVINO_OP_REG(AvgPool, ngraph::op::v1)
_OPENVINO_OP_REG(BatchNormInference, ngraph::op::v0)
_OPENVINO_OP_REG(BatchToSpace, ngraph::op::v1)
_OPENVINO_OP_REG(BinaryConvolution, ngraph::op::v1)
_OPENVINO_OP_REG(Broadcast, ngraph::op::v1)
_OPENVINO_OP_REG(Broadcast, ngraph::op::v3)
_OPENVINO_OP_REG(Bucketize, ngraph::op::v3)
_OPENVINO_OP_REG(CTCGreedyDecoder, ngraph::op::v0)
_OPENVINO_OP_REG(Ceiling, ngraph::op::v0)
_OPENVINO_OP_REG(Clamp, ngraph::op::v0)
_OPENVINO_OP_REG(Concat, ngraph::op::v0)
_OPENVINO_OP_REG(Constant, ngraph::op::v0)
_OPENVINO_OP_REG(Convert, ngraph::op::v0)
_OPENVINO_OP_REG(ConvertLike, ngraph::op::v1)
_OPENVINO_OP_REG(Convolution, ngraph::op::v1)
_OPENVINO_OP_REG(ConvolutionBackpropData, ngraph::op::v1)
_OPENVINO_OP_REG(Cos, ngraph::op::v0)
_OPENVINO_OP_REG(Cosh, ngraph::op::v0)
_OPENVINO_OP_REG(CumSum, ngraph::op::v0)
_OPENVINO_OP_REG(DeformableConvolution, ngraph::op::v1)
_OPENVINO_OP_REG(DeformablePSROIPooling, ngraph::op::v1)
_OPENVINO_OP_REG(DepthToSpace, ngraph::op::v0)
_OPENVINO_OP_REG(DetectionOutput, ngraph::op::v0)
_OPENVINO_OP_REG(Divide, ngraph::op::v1)
_OPENVINO_OP_REG(Elu, ngraph::op::v0)
_OPENVINO_OP_REG(EmbeddingBagOffsetsSum, ngraph::op::v3)
_OPENVINO_OP_REG(EmbeddingBagPackedSum, ngraph::op::v3)
_OPENVINO_OP_REG(EmbeddingSegmentsSum, ngraph::op::v3)
_OPENVINO_OP_REG(Equal, ngraph::op::v1)
_OPENVINO_OP_REG(Erf, ngraph::op::v0)
_OPENVINO_OP_REG(Exp, ngraph::op::v0)
_OPENVINO_OP_REG(ExtractImagePatches, ngraph::op::v3)
_OPENVINO_OP_REG(FakeQuantize, ngraph::op::v0)
_OPENVINO_OP_REG(Floor, ngraph::op::v0)
_OPENVINO_OP_REG(FloorMod, ngraph::op::v1)
_OPENVINO_OP_REG(GRN, ngraph::op::v0)
_OPENVINO_OP_REG(GRUCell, ngraph::op::v3)
_OPENVINO_OP_REG(Gather, ngraph::op::v1)
_OPENVINO_OP_REG(GatherND, ngraph::op::v5)
_OPENVINO_OP_REG(GatherTree, ngraph::op::v1)
_OPENVINO_OP_REG(Gelu, ngraph::op::v0)
_OPENVINO_OP_REG(Greater, ngraph::op::v1)
_OPENVINO_OP_REG(GreaterEqual, ngraph::op::v1)
_OPENVINO_OP_REG(GroupConvolution, ngraph::op::v1)
_OPENVINO_OP_REG(GroupConvolutionBackpropData, ngraph::op::v1)
_OPENVINO_OP_REG(GroupNormalization, ngraph::op::v12)
_OPENVINO_OP_REG(HardSigmoid, ngraph::op::v0)
_OPENVINO_OP_REG(Interpolate, ngraph::op::v0)
_OPENVINO_OP_REG(Interpolate, ngraph::op::v4)
_OPENVINO_OP_REG(LRN, ngraph::op::v0)
_OPENVINO_OP_REG(LSTMCell, ngraph::op::v0)
_OPENVINO_OP_REG(LSTMSequence, ngraph::op::v0)
_OPENVINO_OP_REG(Less, ngraph::op::v1)
_OPENVINO_OP_REG(LessEqual, ngraph::op::v1)
_OPENVINO_OP_REG(Log, ngraph::op::v0)
_OPENVINO_OP_REG(LogicalAnd, ngraph::op::v1)
_OPENVINO_OP_REG(LogicalNot, ngraph::op::v1)
_OPENVINO_OP_REG(LogicalOr, ngraph::op::v1)
_OPENVINO_OP_REG(LogicalXor, ngraph::op::v1)
_OPENVINO_OP_REG(MVN, ngraph::op::v0)
_OPENVINO_OP_REG(MatMul, ngraph::op::v0)
_OPENVINO_OP_REG(MaxPool, ngraph::op::v1)
_OPENVINO_OP_REG(Maximum, ngraph::op::v1)
_OPENVINO_OP_REG(Minimum, ngraph::op::v1)
_OPENVINO_OP_REG(Mod, ngraph::op::v1)
_OPENVINO_OP_REG(Multiply, ngraph::op::v1)
_OPENVINO_OP_REG(Negative, ngraph::op::v0)
_OPENVINO_OP_REG(NonMaxSuppression, ngraph::op::v1)
_OPENVINO_OP_REG(NonMaxSuppression, ngraph::op::v3)
_OPENVINO_OP_REG(NonZero, ngraph::op::v3)
_OPENVINO_OP_REG(NormalizeL2, ngraph::op::v0)
_OPENVINO_OP_REG(NotEqual, ngraph::op::v1)
_OPENVINO_OP_REG(OneHot, ngraph::op::v1)
_OPENVINO_OP_REG(PRelu, ngraph::op::v0)
_OPENVINO_OP_REG(PSROIPooling, ngraph::op::v0)
_OPENVINO_OP_REG(Pad, ngraph::op::v1)
_OPENVINO_OP_REG(Parameter, ngraph::op::v0)
_OPENVINO_OP_REG(Power, ngraph::op::v1)
_OPENVINO_OP_REG(PriorBox, ngraph::op::v0)
_OPENVINO_OP_REG(PriorBoxClustered, ngraph::op::v0)
_OPENVINO_OP_REG(Proposal, ngraph::op::v0)
_OPENVINO_OP_REG(RNNCell, ngraph::op::v0)
_OPENVINO_OP_REG(ROIPooling, ngraph::op::v0)
_OPENVINO_OP_REG(Range, ngraph::op::v0)
_OPENVINO_OP_REG(ReduceMax, ngraph::op::v1)
_OPENVINO_OP_REG(ReduceLogicalAnd, ngraph::op::v1)
_OPENVINO_OP_REG(ReduceLogicalOr, ngraph::op::v1)
_OPENVINO_OP_REG(ReduceMean, ngraph::op::v1)
_OPENVINO_OP_REG(ReduceMin, ngraph::op::v1)
_OPENVINO_OP_REG(ReduceProd, ngraph::op::v1)
_OPENVINO_OP_REG(ReduceSum, ngraph::op::v1)
_OPENVINO_OP_REG(RegionYolo, ngraph::op::v0)
_OPENVINO_OP_REG(Relu, ngraph::op::v0)
_OPENVINO_OP_REG(ReorgYolo, ngraph::op::v0)
_OPENVINO_OP_REG(Reshape, ngraph::op::v1)
_OPENVINO_OP_REG(Result, ngraph::op::v0)
_OPENVINO_OP_REG(Reverse, ngraph::op::v1)
_OPENVINO_OP_REG(ReverseSequence, ngraph::op::v0)
_OPENVINO_OP_REG(Round, ngraph::op::v5)
_OPENVINO_OP_REG(ROIAlign, ngraph::op::v3)
_OPENVINO_OP_REG(ScatterElementsUpdate, ngraph::op::v3)
_OPENVINO_OP_REG(ScatterUpdate, ngraph::op::v3)
_OPENVINO_OP_REG(Select, ngraph::op::v1)
_OPENVINO_OP_REG(Selu, ngraph::op::v0)
_OPENVINO_OP_REG(ShapeOf, ngraph::op::v0)
_OPENVINO_OP_REG(ShapeOf, ngraph::op::v3)
_OPENVINO_OP_REG(ShuffleChannels, ngraph::op::v0)
_OPENVINO_OP_REG(Sigmoid, ngraph::op::v0)
_OPENVINO_OP_REG(Sign, ngraph::op::v0)
_OPENVINO_OP_REG(Sin, ngraph::op::v0)
_OPENVINO_OP_REG(Sinh, ngraph::op::v0)
_OPENVINO_OP_REG(Softmax, ngraph::op::v1)
_OPENVINO_OP_REG(SpaceToBatch, ngraph::op::v1)
_OPENVINO_OP_REG(SpaceToDepth, ngraph::op::v0)
_OPENVINO_OP_REG(Split, ngraph::op::v1)
_OPENVINO_OP_REG(Sqrt, ngraph::op::v0)
_OPENVINO_OP_REG(SquaredDifference, ngraph::op::v0)
_OPENVINO_OP_REG(Squeeze, ngraph::op::v0)
_OPENVINO_OP_REG(StridedSlice, ngraph::op::v1)
_OPENVINO_OP_REG(Subtract, ngraph::op::v1)
_OPENVINO_OP_REG(Tan, ngraph::op::v0)
_OPENVINO_OP_REG(Tanh, ngraph::op::v0)
_OPENVINO_OP_REG(TensorIterator, ngraph::op::v0)
_OPENVINO_OP_REG(Tile, ngraph::op::v0)
_OPENVINO_OP_REG(TopK, ngraph::op::v1)
_OPENVINO_OP_REG(Transpose, ngraph::op::v1)
_OPENVINO_OP_REG(Unsqueeze, ngraph::op::v0)
_OPENVINO_OP_REG(VariadicSplit, ngraph::op::v1)
_OPENVINO_OP_REG(Xor, ngraph::op::v0)
_OPENVINO_OP_REG(Assign, ngraph::op::v3)
_OPENVINO_OP_REG(ReadValue, ngraph::op::v3)
