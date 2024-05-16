#ifndef MLIR_TRANSFORMS_GRAPHPARTITION_H_
#define MLIR_TRANSFORMS_GRAPHPARTITION_H_

#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/TypeID.h"
#include <cstdint>
#ifndef SIZE_TRANSFORM_PARAMETER
#define SIZE_TRANSFORM_PARAMETER 0.8
#endif

#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace mlir {
class Pass;
class Value;

#define GEN_PASS_DECL_GRAPHPARTITION
#include "mlir/Transforms/Passes.h.inc"

/// Class for blocks after eliminating control flow.
class OpNode {
public:
  /// Add a basic block to the `OpNode`.
  void addBasicBlock(Block *block) { basicBlocks.push_back(block); };

  /// Calculate the block costs and add them to timingCost and spaceCost.
  /// This only calculates the cost of one block in the node.
  void calculateCost(Block *, CFGLoopInfo &loopInfo, int64_t loopTimes,
                     int64_t freeCycle, int64_t basicCycle,
                     int64_t expensiveCycle);

  void setLiveIn(const SmallPtrSet<Value, 16> &values) { inValues = values; }
  void setLiveOut(const SmallPtrSet<Value, 16> &values) { outValues = values; }

  int64_t getTimingCost() { return timingCost; }
  int64_t getSpaceCost() { return spaceCost; }

  SmallPtrSet<Value, 16> getInValues() { return inValues; }
  SmallPtrSet<Value, 16> getOutValues() { return outValues; }

  SmallVector<Block *, 8> getBasicBlocks() { return basicBlocks; }

  bool emptyBasicBlocks() { return basicBlocks.empty(); }

private:
  void calculateSpaceCost(Block *block);
  // use -loop-times
  void calculateTimingCost(Block *block, CFGLoopInfo &loopInfo,
                           int64_t loopTimes, int64_t freeCycle,
                           int64_t basicCycle, int64_t expensiveCycle);

private:
  /// Control flow nodes containing minimum blocks without branches.
  SmallVector<Block *, 8> basicBlocks;

  /// The set of values that are live at the entry of the node.
  SmallPtrSet<Value, 16> inValues;

  /// The set of values that are live at the exit of the node.
  SmallPtrSet<Value, 16> outValues;

  /// Cost of dynamic instructions.
  int64_t timingCost = 0;

  /// Cost of static instructions.
  int64_t spaceCost = 0;
};

enum class GraphKInd {
  TopGraph, // Top-level graph, containing several subgraphs.
  Subgraph  // The subgraph, containing several `OpNode`s.
};

/// BasicGraph has two types, top graph and subgraph.
/// -Top graph contains several subgraphs that kind are `Graph`.
/// -Subgraph contains several `OpNode`s.
class Graph {

public:
  Graph(GraphKInd type) : graphType(type) {}

  /// Returns true if this graph represents an `TopGrpah`.
  bool isTopGraph() const { return graphType == GraphKInd::TopGraph; }

  /// Returns true if this graph represents an `Subgraph`.
  bool isSubgraph() const { return graphType == GraphKInd::Subgraph; }

  SmallVector<OpNode *, 32> getOpNodes() { return opNodes; }
  SmallVector<std::tuple<OpNode *, OpNode *>, 64> getOpNodeEdge() {
    return opNodeEdges;
  }

  int64_t getTimingCost() { return timingCost; }
  int64_t getSpaceCost() { return spaceCost; }

  void addOpNodeAndEdge(OpNode *node, const SmallPtrSet<Value, 16> &values);
  void popOpNodeEdge() { opNodeEdges.pop_back(); }

private:
  GraphKInd graphType;

  SmallVector<OpNode *, 32> opNodes;
  SmallVector<Graph *, 32> subgraphs;

  /// Represent edges between `opNode`s, the tuple consists of the source node
  /// and the destination node in sequence.
  SmallVector<std::tuple<OpNode *, OpNode *>, 64> opNodeEdges;
  /// Represent edges between `Graph`s, the tuple consists of the source graph
  /// and the destination graph in sequence.
  SmallVector<std::tuple<Graph *, Graph *>, 64> subgraphEdges;

  /// Cost of dynamic instructions.
  int64_t timingCost = 0;

  /// Cost of static instructions.
  int64_t spaceCost = 0;
};

/// Creates a pass to subgraph op graphs.
std::unique_ptr<Pass> createGraphPartitionPass(raw_ostream &os = llvm::errs());

} // namespace mlir

#endif // MLIR_TRANSFORMS_GRAPHPARTITION_H_
