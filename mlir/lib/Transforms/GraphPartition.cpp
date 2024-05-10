#include "mlir/Transforms/GraphPartition.h"

#include "mlir-c/IR.h"
#include "mlir/Analysis/Liveness.h"
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
#include "llvm/Analysis/CostModel.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IntrinsicsMips.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
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

    PassManager pm(op->getName());

    pm.addPass(createInlinerPass());

    if (!mlir::succeeded(pm.run(op))) {
      op->emitOpError() << "pipeline fail";
    }

    bool foundMainFunction = false;
    op->walk([&](LLVM::LLVMFuncOp func) {
      if (func.getName() != mainFunctionName) {
        func.setPrivate();
      } else {
        foundMainFunction = true;
      }
    });
    if (!foundMainFunction) {
      mlir::emitError(op->getLoc(),
                      "Main function " + mainFunctionName + " not found!\n");
      return;
    }

    pm.addPass(createCSEPass());
    pm.addPass(createControlFlowSinkPass());
    pm.addPass(createSCCPPass());
    pm.addPass(createSymbolDCEPass());

    if (!mlir::succeeded(pm.run(op))) {
      op->emitOpError() << "pipeline fail";
    }

    // function---------------------------------//

    SymbolTable symbolTable(op);

    // 也许可以利用这个在 inline 后找到 main 函数分析，无需删除其他函数
    // LLVM::LLVMFuncOp mainFunc =
    //     symbolTable.lookup<LLVM::LLVMFuncOp>(mainFunctionName);

    // 打印指定function的op graph，以dot形式存储在文件中
    // mainFunc->getRegion(0).viewGraph();

    llvm::errs() << "after-getNumRegions:" << op->getNumRegions() << "\n";

    // -----------------------------------------//

    Liveness live(op);

    op->walk([&](Block *block) {
      DominanceInfo domInfo;
      PostDominanceInfo postDomInfo;
      CFNode cfn;

      auto num = block->getOperations().size();
      llvm::errs() << "code_size:" << num << "\n";

      blockToId[block] = numBlock++;

      for (auto *suc : block->getSuccessors()) {
        blockEdges.emplace_back(block, suc);
      }

      if (visitedBlock[block])
        return;

      addBlock(block, cfn, domInfo, postDomInfo);

      // Don't include the end block that is in the next node.
      cfNodes.push_back(cfn);

      // llvm::errs() << "******************************\n";

      // // block->dump();
      // // llvm::errs() << "\n";

      // // for (auto value : live.getLiveIn(block)) {
      // //   value.dump();
      // //   llvm::errs() << "\n";
      // // }

      // const auto &liveInValues = live.getLiveIn(block);
      // // node.setLiveIn()

      // llvm::errs() << "num-live-in-value:" << liveInValues.size() << "\n";
      // llvm::errs() << "num-block:" << cfn.size() << "\n";

      // llvm::errs() << "******************************\n";
    });

    // The last node is the whole function that needs to be popped.
    cfNodes.pop_back();
    // TODO: Handling redundant blocks instead of popping them up.

    // printControlFlowNodes();
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

void Node::calculateCost(Block *block) {

}

void Node::calculateSpaceCost(Block *block) {
  spaceCost = block->getOperations().size() * SIZE_TRANSFORM_PARAMETER;
}

void Node::calculateTimingCost(Block *block) {
  timingCost = 0;
  block->walk([&](Operation *op) {
    TargetCostConstants operationCost = TypeSwitch<Operation *, TargetCostConstants>(op)
    .Case([](LLVM::AShrOp op){return TCC_Basic;})
    .Case([](LLVM::AddOp op){return TCC_Basic;})
    .Case([](LLVM::AddrSpaceCastOp op){return TCC_Basic;})
    .Case([](LLVM::AddressOfOp op){return TCC_Basic;})
    .Case([](LLVM::AllocaOp op){return TCC_Basic;})
    .Case([](LLVM::AndOp op){return TCC_Basic;})
    .Case([](LLVM::AtomicCmpXchgOp op){return TCC_Basic;})
    .Case([](LLVM::AtomicRMWOp op){return TCC_Basic;})
    .Case([](LLVM::BitcastOp op){return TCC_Basic;})
    .Case([](LLVM::BrOp op){return TCC_Basic;})
    .Case([](LLVM::CallIntrinsicOp op){return TCC_Basic;})
    .Case([](LLVM::CallOp op){return TCC_Basic;})
    .Case([](LLVM::ComdatOp op){return TCC_Basic;})
    .Case([](LLVM::ComdatSelectorOp op){return TCC_Basic;})
    .Case([](LLVM::CondBrOp op){return TCC_Basic;})
    .Case([](LLVM::ConstantOp op){return TCC_Basic;})
    .Case([](LLVM::ExtractElementOp op){return TCC_Basic;})
    .Case([](LLVM::ExtractValueOp op){return TCC_Basic;})
    .Case([](LLVM::FCmpOp op){return TCC_Basic;})
    .Case([](LLVM::FDivOp op){return TCC_Basic;})
    .Case([](LLVM::FMulOp op){return TCC_Basic;})
    .Case([](LLVM::FNegOp op){return TCC_Basic;})
    .Case([](LLVM::FPExtOp op){return TCC_Basic;})
    .Case([](LLVM::FPToSIOp op){return TCC_Basic;})
    .Case([](LLVM::FPToUIOp op){return TCC_Basic;})
    .Case([](LLVM::FPTruncOp op){return TCC_Basic;})
    .Case([](LLVM::FSubOp op){return TCC_Basic;})
    .Case([](LLVM::FenceOp op){return TCC_Basic;})
    .Case([](LLVM::FreezeOp op){return TCC_Basic;})
    ;
    timingCost += operationCost;});   
}

std::unique_ptr<Pass> mlir::createGraphPartitionPass(raw_ostream &os) {
  return std::make_unique<PartitionPass>(os);
}