#include "mlir/Transforms/GraphPartition.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <ostream>
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_GRAPHPARTITION
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

class PartitionPass : public impl::GraphPartitionBase<PartitionPass> {
  using CFNode = std::vector<Block *>;

public:
  PartitionPass(raw_ostream &os) : os(os) {}
  PartitionPass(const PartitionPass &o) : PartitionPass(o.os.getOStream()) {}

  Block *addBlock(Block *block, CFNode &cfn, DominanceInfo &domInfo,
                  PostDominanceInfo &postDomInfo) {
    // auto *rootBlock = cfn.front();
    Block *preEndBlock = nullptr;
    Block *sucEndBlock = nullptr;

    auto it = std::find(cfn.begin(), cfn.end(), block);
    if (it != cfn.end()) {
      return nullptr;
    }

    // Add block to CFNode.
    cfn.push_back(block);
    if (block->getNumSuccessors() > 1) {
      visitedCondBrBlock.push_back(block);
    }

    auto *rootBlock = cfn.front();
    auto isDom = domInfo.properlyDominates(rootBlock, block);
    auto isPostDom = postDomInfo.properlyPostDominates(block, rootBlock);

    if (isDom && isPostDom) {
      return block;
    }
    // if (!isDom) {
    //   for (auto *pre : block->getPredecessors()) {
    //     if (auto *eb = addBlock(pre, cfn, domInfo, postDomInfo)) {
    //       preEndBlock = eb;
    //       // TODO: Results may not be the same for each loop.
    //     }
    //   }
    // }
    if (!isPostDom) {
      for (auto *suc : block->getSuccessors()) {
        if (auto *eb = addBlock(suc, cfn, domInfo, postDomInfo)) {
          sucEndBlock = eb;
          // TODO: Results may not be the same for each loop.
        }
      }
    }

    // if (preEndBlock != sucEndBlock) {
    //   llvm::errs() << "End block search failed.\n";
    //   return nullptr;
    // }
    return sucEndBlock;
  }

  void runOnOperation() override {
    Operation *op = getOperation();

    // PassManager pm(op->getName());

    // pm.addPass(createInlinerPass());
    // pm.addPass(createCSEPass());
    // pm.addPass(createControlFlowSinkPass());
    // pm.addPass(createSCCPPass());
    // pm.addPass(createSymbolDCEPass());

    // if (!mlir::succeeded(pm.run(op))) {
    //   op->emitOpError() << "pipeline fail";
    // }

    // auto blocks = op->getBlockOperands();

    // auto *node = domInfo.getNode(op->getBlock());
    // node->getBlock()->dump();
    // auto *dt = df.getRootNode(op->getRegions().end());
    // auto *node = df.getNode(op->getBlock());
    // mlir::emitError(op->getLoc(), std::to_string(node->getNumChildren()));
    // node->getBlock()->dump();

    op->walk([&](Operation *op) {
      DominanceInfo domInfo(op);
      PostDominanceInfo postDomInfo(op);

      if (llvm::isa<LLVM::CondBrOp>(op)) {
        Block *block = op->getBlock();
        CFNode cfn;

        auto it = std::find(visitedCondBrBlock.begin(),
                            visitedCondBrBlock.end(), block);
        if (it != visitedCondBrBlock.end())
          return;

        visitedCondBrBlock.push_back(block);

        auto *endBlock = addBlock(block, cfn, domInfo, postDomInfo);

        // llvm::errs() << "endBlock:\n";
        // endBlock->dump();

        cfNodes.push_back(cfn);
      }

      // if (llvm::isa<BranchOpInterface>(op)) {
      //   op->dump();
      //   if (llvm::isa<LLVM::CondBrOp>(op)) {
      //     for (auto *sc : op->getSuccessors()) {
      //       sc->dump();
      //     }
      //   }
      //   if (llvm::isa<LLVM::BrOp>(op)) {
      //     for (auto *sc : op->getSuccessors()) {
      //       sc->dump();

      //       if (sc->getNumSuccessors() == 1) {
      //         sc->getNextNode()->dump();
      //       }
      //     }
      //   }
      // }
    });

    llvm::errs() << "cfNodes_num:" << cfNodes.size() << "\n";
    for (const auto &cfNode : cfNodes) {
      llvm::errs() << "num:" << cfNode.size() << "\n";
      // for (auto *block : cfNode) {
      //   block->dump();
      // }
    }
  }

private:
  /// Output stream to write DOT file to.
  raw_indented_ostream os;

  /// A structure for storing control flow nodes.
  std::vector<CFNode> cfNodes;

  // DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
  // PostDominanceInfo &postDomInfo = getAnalysis<PostDominanceInfo>();

  std::vector<Block *> visitedCondBrBlock;
};

} // namespace

std::unique_ptr<Pass> mlir::createGraphPartitionPass(raw_ostream &os) {
  return std::make_unique<PartitionPass>(os);
}