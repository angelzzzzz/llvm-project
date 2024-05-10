#ifndef MLIR_TRANSFORMS_GRAPHPARTITION_H_
#define MLIR_TRANSFORMS_GRAPHPARTITION_H_

#ifndef SIZE_TRANSFORM_PARAMETER
#define SIZE_TRANSFORM_PARAMETER 3.2
#endif

#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
class Pass;
class Value;

#define GEN_PASS_DECL_GRAPHPARTITION
#include "mlir/Transforms/Passes.h.inc"

// Class for blocks after eliminate control flow.
class Node {
public:
  void addBasicBlock(Block *);

  // Calculate the block costs and add them to timingCost and spaceCost.
  // This only calculates the cost of one block in the node.
  void calculateCost(Block *);

  void setLiveIn(const SmallPtrSet<Value, 16> &values) { inValues = values; }
  void setLiveOut(const SmallPtrSet<Value, 16> &values) { outValues = values; }

  SmallVector<Block *, 8> getBasicBlocks() { return basicBlocks; }
  int64_t getTimingCost() { return timingCost; }
  int64_t getSpaceCost() { return spaceCost; }

private:
  void calculateSpaceCost(Block *);
  // use -loop-times
  void calculateTimingCost(Block *);

private:
  // Control flow nodes containing minimum blocks without branches.
  SmallVector<Block *, 8> basicBlocks;
  // dynamic instructions
  unsigned long timingCost;
  // static instructions
  unsigned long spaceCost;
  // The set of values that are live at the entry of the node.
  SmallPtrSet<Value, 16> inValues;
  // The set of values that are live at the exit of the node.
  SmallPtrSet<Value, 16> outValues;
  // Underlying constants for 'cost' values in this interface.
  enum TargetCostConstants {
  TCC_Free = 0,     ///< Expected to fold away in lowering.
  TCC_Basic = 1,    ///< The cost of a typical 'add' instruction.
  TCC_Expensive = 4 ///< The cost of a 'div' instruction on x86.
  };
};

// The subgraph of graph, containing several nodes.
class Subgraph {
public:
  SmallVector<Node *, 16> getNodes() { return nodes; }

  void addNode(Node *node);
  void removeNode(Node *node);

private:
  SmallVector<Node *, 16> nodes;
  unsigned long timingCost;
  unsigned long spaceCost;
};

class Graph {
public:
  void addNode(Node *node);
  void addSubgraph(Subgraph *subgraph);

  // Move the node from its local subgraph to the target subgraph
  void moveNode(Node *node, Subgraph *targetSubgraph);

  Subgraph *getSubgraph(Node *node);

private:
  SmallVector<Node *, 64> nodes;
  SmallVector<Subgraph *, 64> subgraphs;

  // Represent edges between nodes, the tuple consists of the source node and
  // the destination node in sequence.
  SmallVector<std::tuple<Node *, Node *>, 128> edges;
};

/// Creates a pass to subgraph op graphs.
std::unique_ptr<Pass> createGraphPartitionPass(raw_ostream &os = llvm::errs());

} // namespace mlir

#endif // MLIR_TRANSFORMS_GRAPHPARTITION_H_
