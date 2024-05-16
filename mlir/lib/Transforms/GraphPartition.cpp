#include "mlir/Transforms/GraphPartition.h"

#include "mlir-c/IR.h"
#include "mlir/Analysis/CFGLoopInfo.h"
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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IntrinsicsMips.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
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
  using BlockEdge = std::tuple<Block *, Block *>;

public:
  PartitionPass(raw_ostream &os) : os(os) {}
  PartitionPass(const PartitionPass &o) : PartitionPass(o.os.getOStream()) {}

  /// Add blocks based on dominant information to eliminate control flow.
  /// Add the begin block and its successors to the node.
  void addBlock(Block *block, OpNode *node, DominanceInfo &domInfo,
                PostDominanceInfo &postDomInfo, CFGLoopInfo &loopInfo) {

    if (visitedBlock[block])
      return;

    // Analyze dominance and post-dominance information between the begin block
    // and the current block.
    Block *beginBlock;
    if (node->emptyBasicBlocks()) {
      beginBlock = block;
    } else {
      beginBlock = node->getBasicBlocks().front();
    }
    auto isDom = domInfo.properlyDominates(beginBlock, block);
    auto isPostDom = postDomInfo.properlyPostDominates(block, beginBlock);

    // The block is the end block of node, exit recursion.
    if (isDom && isPostDom) {
      return;
    }

    // Add block to OpNode.
    node->addBasicBlock(block);

    visitedBlock[block] = true;

    //  Calculate the space and timing costs of the current block, and add them
    //  to the cost of the node.
    node->calculateCost(block, loopInfo, loopTimes, freeCycleCost,
                        basicCycleCost, expensiveCycleCost);

    // The block is not the end block, so we need to traverse successors.
    for (auto *suc : block->getSuccessors()) {
      addBlock(suc, node, domInfo, postDomInfo, loopInfo);
    }
  }

  // void printControlFlowNodes() {
  //   int index = 1;

  //   os << "digraph G {\n";
  //   os.indent();

  //   for (const auto &cfNode : cfNodes) {
  //     os << "subgraph cluster_" << index << " {\n";
  //     os.indent();
  //     os << "label=\"Node " << index << "\";\n";
  //     os << "color=red;\n";
  //     for (auto *block : cfNode) {
  //       os << "v" << blockToId[block] << " [label=\"bb" << blockToId[block]
  //          << "\"];\n";
  //     }
  //     os.unindent();
  //     os << "}\n";
  //     index++;
  //   }

  //   for (auto edge : blockEdges) {
  //     os << "v" << blockToId[std::get<0>(edge)] << " -> v"
  //        << blockToId[std::get<1>(edge)] << ";\n";
  //   }

  //   os.unindent();
  //   os << "}\n";
  // }

  void runOnOperation() override {
    Operation *op = getOperation();

    PassManager pm(op->getName());

    pm.addPass(createInlinerPass());
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
    LLVM::LLVMFuncOp mainFunc =
        symbolTable.lookup<LLVM::LLVMFuncOp>(mainFunctionName);
    if (!mainFunc) {
      mlir::emitError(op->getLoc(),
                      "Main function " + mainFunctionName + " not found!\n");
      return;
    }

    // 打印指定function的op graph，以dot形式存储在文件中
    // mainFunc->getRegion(0).viewGraph();

    // -----------------------------------------//

    // The highest level graph with several subgraphs.
    Graph topGraph(GraphKInd::TopGraph);
    // When initializing the graph, all nodes are in the same subgraph.
    Graph subgraph(GraphKInd::Subgraph);
    // Get live information of op.
    Liveness live(mainFunc);
    // Used to analyze dominant information.
    DominanceInfo domInfo;
    PostDominanceInfo postDomInfo;

    // Used to get blocks in all loop bodies.
    llvm::DominatorTreeBase<Block, false> &domTree =
        domInfo.getDomTree(&mainFunc->getRegion(0));
    CFGLoopInfo loopInfo(domTree);

    mainFunc->walk([&](Block *block) {
      blockToId[block] = numBlock++;

      // 处理Block间的边，是否需要？用于dot
      for (auto *suc : block->getSuccessors()) {
        blockEdges.emplace_back(block, suc);
      }

      if (visitedBlock[block])
        return;

      OpNode *node = new OpNode;

      addBlock(block, node, domInfo, postDomInfo, loopInfo);

      subgraph.addOpNodeAndEdge(node, live.getLiveIn(block));
    });

    // Remove the last edge which doesn't have a destination.
    subgraph.popOpNodeEdge();

    // Print graph after initialization
    llvm::errs() << "node-num:" << subgraph.getOpNodes().size() << "\n";
    llvm::errs() << "edge-num:" << subgraph.getOpNodeEdge().size() << "\n";
    llvm::errs() << "space:" << subgraph.getSpaceCost() << "\n";
    llvm::errs() << "timing:" << subgraph.getTimingCost() << "\n";
    for (auto *n : subgraph.getOpNodes()) {
      llvm::errs() << "--------------------\n";
      llvm::errs() << "space:" << n->getSpaceCost() << "\n";
      llvm::errs() << "timing:" << n->getTimingCost() << "\n";
      llvm::errs() << "num-blocks:" << n->getBasicBlocks().size() << "\n";
      llvm::errs() << "num-in-values:" << n->getInValues().size() << "\n";
      llvm::errs() << "num-out-values:" << n->getOutValues().size() << "\n";
      llvm::errs() << "--------------------\n";
    }
  }

private:
  MLIRContext context;

  /// Output stream to write DOT codes.
  raw_indented_ostream os;

  std::vector<BlockEdge> blockEdges;

  DenseMap<Block *, bool> visitedBlock;

  /// Used to generate DOT for graph.
  DenseMap<Block *, int> blockToId;

  int numBlock = 0;

  std::vector<LLVM::LLVMFuncOp> calleeList;
};

} // namespace

void OpNode::calculateCost(Block *block, CFGLoopInfo &loopInfo,
                           int64_t loopTimes, int64_t freeCycle,
                           int64_t basicCycle, int64_t expensiveCycle) {
  calculateSpaceCost(block);
  calculateTimingCost(block, loopInfo, loopTimes, freeCycle, basicCycle,
                      expensiveCycle);
}

void OpNode::calculateSpaceCost(Block *block) {
  spaceCost += block->getOperations().size() * SIZE_TRANSFORM_PARAMETER;
}

void OpNode::calculateTimingCost(Block *block, CFGLoopInfo &loopInfo,
                                 int64_t loopTimes, int64_t freeCycle,
                                 int64_t basicCycle, int64_t expensiveCycle) {
  int64_t blockTimingCost = 0;
  block->walk([&](Operation *op) {
    int64_t operationCost =
        TypeSwitch<Operation *, unsigned>(op)
            .Case<LLVM::CallIntrinsicOp, LLVM::CallOp, LLVM::ComdatOp,
                  LLVM::ComdatSelectorOp, LLVM::FreezeOp, LLVM::LLVMFuncOp,
                  LLVM::NoneTokenOp, LLVM::PoisonOp, LLVM::UndefOp,
                  LLVM::UnreachableOp, LLVM::ZeroOp, LLVM::Annotation>(
                [&](auto op) { return freeCycle; })
            .Case<
                LLVM::AShrOp, LLVM::AddOp, LLVM::AddrSpaceCastOp,
                LLVM::AddressOfOp, LLVM::AllocaOp, LLVM::AndOp, LLVM::BitcastOp,
                LLVM::BrOp, LLVM::CondBrOp, LLVM::ConstantOp,
                LLVM::ExtractElementOp, LLVM::ExtractValueOp, LLVM::FCmpOp,
                LLVM::FMulOp, LLVM::FNegOp, LLVM::FPExtOp, LLVM::FPToSIOp,
                LLVM::FPToUIOp, LLVM::FPTruncOp, LLVM::FSubOp, LLVM::FenceOp,
                LLVM::GEPOp, LLVM::GlobalOp, LLVM::ICmpOp,
                LLVM::InsertElementOp, LLVM::InsertValueOp, LLVM::InsertValueOp,
                LLVM::IntToPtrOp, LLVM::LShrOp, LLVM::LinkerOptionsOp,
                LLVM::MulOp, LLVM::OrOp, LLVM::PtrToIntOp, LLVM::ReturnOp,
                LLVM::SExtOp, LLVM::SIToFPOp, LLVM::SelectOp, LLVM::ShlOp,
                LLVM::ShuffleVectorOp, LLVM::SubOp, LLVM::SwitchOp,
                LLVM::TruncOp, LLVM::XOrOp, LLVM::ZExtOp, LLVM::AbsOp,
                LLVM::AssumeOp, LLVM::AssumeOp, LLVM::ByteSwapOp,
                LLVM::CopySignOp, LLVM::CoroAlignOp, LLVM::CoroEndOp,
                LLVM::CoroFreeOp, LLVM::CoroIdOp, LLVM::CoroPromiseOp,
                LLVM::CoroSaveOp, LLVM::CoroSizeOp, LLVM::CountLeadingZerosOp,
                LLVM::CountTrailingZerosOp, LLVM::CtPopOp, LLVM::DbgDeclareOp,
                LLVM::DbgLabelOp, LLVM::DbgValueOp, LLVM::EhTypeidForOp,
                LLVM::ExpectOp, LLVM::ExpectWithProbabilityOp, LLVM::FAbsOp,
                LLVM::FCeilOp, LLVM::FFloorOp, LLVM::FMAOp, LLVM::FMulAddOp,
                LLVM::FTruncOp, LLVM::FshlOp, LLVM::FshrOp,
                LLVM::GetActiveLaneMaskOp, LLVM::InvariantEndOp,
                LLVM::InvariantStartOp, LLVM::IsConstantOp, LLVM::IsFPClass,
                LLVM::IsFPClass, LLVM::LifetimeEndOp, LLVM::LifetimeStartOp,
                LLVM::MatrixTransposeOp, LLVM::MatrixTransposeOp,
                LLVM::MaxNumOp, LLVM::MaximumOp, LLVM::MemcpyInlineOp,
                LLVM::MemsetOp, LLVM::MinNumOp, LLVM::MinimumOp,
                LLVM::NearbyintOp, LLVM::NoAliasScopeDeclOp,
                LLVM::PtrAnnotation, LLVM::SAddSat, LLVM::SAddWithOverflowOp,
                LLVM::SMaxOp, LLVM::SMinOp, LLVM::SSACopyOp, LLVM::SSACopyOp,
                LLVM::SSHLSat, LLVM::SSubSat, LLVM::StackRestoreOp,
                LLVM::StackSaveOp, LLVM::StepVectorOp, LLVM::UMaxOp,
                LLVM::UMinOp, LLVM::USHLSat, LLVM::USubSat, LLVM::VPAShrOp,
                LLVM::VPAddOp, LLVM::VPAndOp, LLVM::VPFNegOp, LLVM::VPFSubOp,
                LLVM::VPIntToPtrOp, LLVM::VPLShrOp, LLVM::VPLoadOp,
                LLVM::VPOrOp, LLVM::VPPtrToIntOp, LLVM::VPPtrToIntOp,
                LLVM::VPSExtOp, LLVM::VPSExtOp, LLVM::VPSRemOp, LLVM::VPShlOp,
                LLVM::VPStoreOp, LLVM::VPStridedLoadOp, LLVM::VPStridedStoreOp,
                LLVM::VPSubOp, LLVM::VPTruncOp, LLVM::VPUDivOp,
                LLVM::VPUIToFPOp, LLVM::VPXorOp, LLVM::VPZExtOp, LLVM::VaCopyOp,
                LLVM::VaEndOp, LLVM::VaStartOp, LLVM::VarAnnotation,
                LLVM::vector_extract, LLVM::vector_insert>(
                [&](auto op) { return basicCycle; })
            .Case<LLVM::MatrixMultiplyOp, LLVM::MemcpyOp, LLVM::MemmoveOp,
                  LLVM::AtomicCmpXchgOp, LLVM::AtomicRMWOp, LLVM::CosOp,
                  LLVM::FDivOp, LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp,
                  LLVM::InlineAsmOp, LLVM::InvokeOp, LLVM::LandingpadOp,
                  LLVM::LoadOp, LLVM::MulOp, LLVM::ResumeOp, LLVM::SDivOp,
                  LLVM::SRemOp, LLVM::StoreOp, LLVM::UDivOp, LLVM::UIToFPOp,
                  LLVM::URemOp, LLVM::BitReverseOp, LLVM::BitReverseOp,
                  LLVM::ConstrainedFPTruncIntr, LLVM::CoroBeginOp,
                  LLVM::CoroResumeOp, LLVM::CoroSuspendOp, LLVM::Exp2Op,
                  LLVM::ExpOp, LLVM::LlrintOp, LLVM::LlroundOp, LLVM::Log10Op,
                  LLVM::Log2Op, LLVM::LogOp, LLVM::LrintOp, LLVM::LroundOp,
                  LLVM::MaskedLoadOp, LLVM::MaskedStoreOp,
                  LLVM::MatrixColumnMajorLoadOp, LLVM::MatrixColumnMajorStoreOp,
                  LLVM::PowIOp, LLVM::PowOp, LLVM::RintOp, LLVM::RoundEvenOp,
                  LLVM::RoundOp, LLVM::SMulWithOverflowOp,
                  LLVM::SMulWithOverflowOp, LLVM::SSubWithOverflowOp,
                  LLVM::SinOp, LLVM::SqrtOp, LLVM::ThreadlocalAddressOp,
                  LLVM::UAddWithOverflowOp, LLVM::UBSanTrap,
                  LLVM::UMulWithOverflowOp, LLVM::USubWithOverflowOp,
                  LLVM::VPFAddOp, LLVM::VPFDivOp, LLVM::VPFMulAddOp,
                  LLVM::VPFMulOp, LLVM::VPFPExtOp, LLVM::VPFPToSIOp,
                  LLVM::VPFPToUIOp, LLVM::VPFPTruncOp, LLVM::VPFRemOp,
                  LLVM::VPFmaOp, LLVM::VPMergeMinOp, LLVM::VPMulOp,
                  LLVM::VPReduceAddOp, LLVM::VPReduceAndOp,
                  LLVM::VPReduceFAddOp, LLVM::VPReduceFMaxOp,
                  LLVM::VPReduceFMinOp, LLVM::VPReduceFMulOp,
                  LLVM::VPReduceMulOp, LLVM::VPReduceOrOp, LLVM::VPReduceSMaxOp,
                  LLVM::VPReduceUMinOp, LLVM::VPReduceXorOp,
                  LLVM::VPReduceSMinOp, LLVM::VPReduceUMaxOp,
                  LLVM::VPReduceUMinOp, LLVM::VPReduceXorOp, LLVM::VPSDivOp,
                  LLVM::VPSIToFPOp, LLVM::VPURemOp, LLVM::masked_compressstore,
                  LLVM::masked_expandload, LLVM::masked_gather,
                  LLVM::masked_scatter, LLVM::vector_reduce_add,
                  LLVM::vector_reduce_and, LLVM::vector_reduce_fadd,
                  LLVM::vector_reduce_fmax, LLVM::vector_reduce_fmaximum,
                  LLVM::vector_reduce_fmin, LLVM::vector_reduce_fminimum,
                  LLVM::vector_reduce_mul, LLVM::vector_reduce_or,
                  LLVM::vector_reduce_smax, LLVM::vector_reduce_smin,
                  LLVM::vector_reduce_xor>(
                [&](auto op) { return expensiveCycle; });
    blockTimingCost += operationCost;
  });
  // If the block is in a loop, multiply it by `loopTimes`.
  if (auto *loop = loopInfo.getLoopFor(block)) {
    blockTimingCost = pow(loopTimes, loop->getLoopDepth()) * blockTimingCost;
  }
  timingCost += blockTimingCost;
}

void Graph::addOpNodeAndEdge(OpNode *node,
                             const SmallPtrSet<Value, 16> &values) {

  // Add edge.
  if (!opNodeEdges.empty()) {
    std::get<1>(opNodeEdges.back()) = node;
    // Set `outValues` for the previous node and set `inValues` for the current
    // node.
    opNodes.back()->setLiveOut(values);
    node->setLiveIn(values);
  }
  opNodeEdges.emplace_back(node, nullptr);
  // Add node.
  opNodes.push_back(node);
  timingCost += node->getTimingCost();
  spaceCost += node->getSpaceCost();
}

std::unique_ptr<Pass> mlir::createGraphPartitionPass(raw_ostream &os) {
  return std::make_unique<PartitionPass>(os);
}