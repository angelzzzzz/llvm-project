#include "mlir/Transforms/GraphPartition.h"

#include "mlir-c/IR.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <ostream>
#include <tuple>
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_GRAPHPARTITION
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

class PartitionPass : public impl::GraphPartitionBase<PartitionPass> {
  using CFNode = std::vector<Block *>;
  using BlockEdge = std::tuple<Block *, Block *>;

public:
  PartitionPass(raw_ostream &os) : os(os) {}
  PartitionPass(const PartitionPass &o) : PartitionPass(o.os.getOStream()) {}

  Block *addBlock(Block *block, CFNode &cfn, DominanceInfo &domInfo,
                  PostDominanceInfo &postDomInfo) {
    Block *endBlock = nullptr;

    if (visitedBlock[block])
      return nullptr;

    // Add block to CFNode.
    cfn.push_back(block);
    visitedBlock[block] = true;

    auto *rootBlock = cfn.front();
    auto isDom = domInfo.properlyDominates(rootBlock, block);
    auto isPostDom = postDomInfo.properlyPostDominates(block, rootBlock);

    if (isDom && isPostDom) {
      cfn.pop_back();
      visitedBlock[block] = false;
      return block;
    }
    if (!isPostDom) {
      for (auto *suc : block->getSuccessors()) {
        if (auto *eb = addBlock(suc, cfn, domInfo, postDomInfo)) {
          endBlock = eb;
          // TODO: Results may not be the same for each loop.
        }
      }
    }

    return endBlock;
  }

  void printControlFlowNodes() {
    int index = 1;

    os << "digraph G {\n";
    os.indent();

    for (const auto &cfNode : cfNodes) {
      os << "subgraph cluster_" << index << " {\n";
      os.indent();
      os << "label=\"Node " << index << "\";\n";
      os << "color=red;\n";
      for (auto *block : cfNode) {
        os << "v" << blockToId[block] << " [label=\"bb" << blockToId[block]
           << "\"];\n";
      }
      os.unindent();
      os << "}\n";
      index++;
    }

    for (auto edge : blockEdges) {
      os << "v" << blockToId[std::get<0>(edge)] << " -> v"
         << blockToId[std::get<1>(edge)] << ";\n";
    }

    os.unindent();
    os << "}\n";
  }

  void runOnOperation() override {
    Operation *op = getOperation();

    op->walk([&](mlir::LLVM::CallOp callOp) {
      // callOp->dump();
      auto callee = callOp.getCalleeAttr();
      auto func = llvm::dyn_cast_or_null<LLVM::LLVMFuncOp>(
          SymbolTable::lookupNearestSymbolFrom(op, callee));
      calleeList.push_back(func);
    });

    PassManager pm(op->getName());

    pm.addPass(createInlinerPass());

    if (!mlir::succeeded(pm.run(op))) {
      op->emitOpError() << "pipeline fail";
    }

    for (auto func : calleeList) {
      func.setPrivate();
    }
    pm.addPass(createCSEPass());
    pm.addPass(createControlFlowSinkPass());
    pm.addPass(createSCCPPass());
    pm.addPass(createSymbolDCEPass());

    if (!mlir::succeeded(pm.run(op))) {
      op->emitOpError() << "pipeline fail";
    }

    op->walk([&](Block *block) {
      DominanceInfo domInfo;
      PostDominanceInfo postDomInfo;
      CFNode cfn;

      blockToId[block] = numBlock++;

      for (auto *suc : block->getSuccessors()) {
        blockEdges.emplace_back(block, suc);
      }

      if (visitedBlock[block])
        return;

      addBlock(block, cfn, domInfo, postDomInfo);
      cfNodes.push_back(cfn);
    });

    cfNodes.pop_back();
    // TODO: Remove the last block.

    printControlFlowNodes();
  }

private:
  MLIRContext context;

  /// Output stream to write DOT codes.
  raw_indented_ostream os;

  /// A structure for storing control flow nodes.
  std::vector<CFNode> cfNodes;

  std::vector<BlockEdge> blockEdges;

  DenseMap<Block *, bool> visitedBlock;

  DenseMap<Block *, int> blockToId;
  int numBlock = 0;

  std::vector<LLVM::LLVMFuncOp> calleeList;
};

} // namespace

std::unique_ptr<Pass> mlir::createGraphPartitionPass(raw_ostream &os) {
  return std::make_unique<PartitionPass>(os);
}