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
    .Case([](LLVM::GEPOp op){return TCC_Basic;})
    .Case([](LLVM::GlobalCtorsOp op){return TCC_Basic;})
    .Case([](LLVM::GlobalDtorsOp op){return TCC_Basic;})
    .Case([](LLVM::GlobalOp op){return TCC_Basic;})
    .Case([](LLVM::ICmpOp op){return TCC_Basic;})
    .Case([](LLVM::InlineAsmOp op){return TCC_Basic;})
    .Case([](LLVM::InsertElementOp op){return TCC_Basic;})
    .Case([](LLVM::InsertValueOp op){return TCC_Basic;})
    .Case([](LLVM::IntToPtrOp op){return TCC_Basic;})
    .Case([](LLVM::InvokeOp op){return TCC_Basic;})
    .Case([](LLVM::LLVMFuncOp op){return TCC_Basic;})
    .Case([](LLVM::LShrOp op){return TCC_Basic;})
    .Case([](LLVM::LandingpadOp op){return TCC_Basic;})
    .Case([](LLVM::LinkerOptionsOp op){return TCC_Basic;})
    .Case([](LLVM::LoadOp op){return TCC_Basic;})
    .Case([](LLVM::MulOp op){return TCC_Basic;})
    .Case([](LLVM::NoneTokenOp op){return TCC_Basic;})
    .Case([](LLVM::OrOp op){return TCC_Basic;})
    .Case([](LLVM::PoisonOp op){return TCC_Basic;})
    .Case([](LLVM::PtrToIntOp op){return TCC_Basic;})
    .Case([](LLVM::ResumeOp op){return TCC_Basic;})
    .Case([](LLVM::ReturnOp op){return TCC_Basic;})
    .Case([](LLVM::SDivOp op){return TCC_Basic;})
    .Case([](LLVM::SExtOp op){return TCC_Basic;})
    .Case([](LLVM::SIToFPOp op){return TCC_Basic;})
    .Case([](LLVM::SRemOp op){return TCC_Basic;})
    .Case([](LLVM::SelectOp op){return TCC_Basic;})
    .Case([](LLVM::ShlOp op){return TCC_Basic;})
    .Case([](LLVM::ShuffleVectorOp op){return TCC_Basic;})
    .Case([](LLVM::StoreOp op){return TCC_Basic;})
    .Case([](LLVM::SubOp op){return TCC_Basic;})
    .Case([](LLVM::SwitchOp op){return TCC_Basic;})
    .Case([](LLVM::TruncOp op){return TCC_Basic;})
    .Case([](LLVM::UDivOp op){return TCC_Basic;})
    .Case([](LLVM::UIToFPOp op){return TCC_Basic;})
    .Case([](LLVM::URemOp op){return TCC_Basic;})
    .Case([](LLVM::UndefOp op){return TCC_Basic;})
    .Case([](LLVM::UnreachableOp op){return TCC_Basic;})
    .Case([](LLVM::XOrOp op){return TCC_Basic;})
    .Case([](LLVM::ZExtOp op){return TCC_Basic;})
    .Case([](LLVM::ZeroOp op){return TCC_Basic;})
    .Case([](LLVM::AbsOp op){return TCC_Basic;})
    .Case([](LLVM::Annotation op){return TCC_Basic;})
    .Case([](LLVM::AssumeOp op){return TCC_Basic;})
    .Case([](LLVM::BitReverseOp op){return TCC_Basic;})
    .Case([](LLVM::ByteSwapOp op){return TCC_Basic;})
    .Case([](LLVM::ConstrainedFPTruncIntr op){return TCC_Basic;})
    .Case([](LLVM::CopySignOp op){return TCC_Basic;})
    .Case([](LLVM::CoroAlignOp op){return TCC_Basic;})
    .Case([](LLVM::CoroBeginOp op){return TCC_Basic;})
    .Case([](LLVM::CoroEndOp op){return TCC_Basic;})
    .Case([](LLVM::CoroFreeOp op){return TCC_Basic;})
    .Case([](LLVM::CoroIdOp op){return TCC_Basic;})
    .Case([](LLVM::CoroPromiseOp op){return TCC_Basic;})
    .Case([](LLVM::CoroResumeOp op){return TCC_Basic;})
    .Case([](LLVM::CoroSaveOp op){return TCC_Basic;})
    .Case([](LLVM::CoroSizeOp op){return TCC_Basic;})
    .Case([](LLVM::CoroSuspendOp op){return TCC_Basic;})
    .Case([](LLVM::CosOp op){return TCC_Basic;})
    .Case([](LLVM::CountLeadingZerosOp op){return TCC_Basic;})
    .Case([](LLVM::CountTrailingZerosOp op){return TCC_Basic;})
    .Case([](LLVM::CtPopOp op){return TCC_Basic;})
    .Case([](LLVM::DbgDeclareOp op){return TCC_Basic;})
    .Case([](LLVM::DbgLabelOp op){return TCC_Basic;})
    .Case([](LLVM::DbgValueOp op){return TCC_Basic;})
    .Case([](LLVM::EhTypeidForOp op){return TCC_Basic;})
    .Case([](LLVM::Exp2Op op){return TCC_Basic;})
    .Case([](LLVM::ExpOp op){return TCC_Basic;})
    .Case([](LLVM::ExpectOp op){return TCC_Basic;})
    .Case([](LLVM::ExpectWithProbabilityOp op){return TCC_Basic;})
    .Case([](LLVM::FAbsOp op){return TCC_Basic;})
    .Case([](LLVM::FCeilOp op){return TCC_Basic;})
    .Case([](LLVM::FFloorOp op){return TCC_Basic;})
    .Case([](LLVM::FMAOp op){return TCC_Basic;})
    .Case([](LLVM::FMulAddOp op){return TCC_Basic;})
    .Case([](LLVM::FTruncOp op){return TCC_Basic;})
    .Case([](LLVM::FshlOp op){return TCC_Basic;})
    .Case([](LLVM::FshrOp op){return TCC_Basic;})
    .Case([](LLVM::GetActiveLaneMaskOp op){return TCC_Basic;})
    .Case([](LLVM::InvariantEndOp op){return TCC_Basic;})
    .Case([](LLVM::InvariantStartOp op){return TCC_Basic;})
    .Case([](LLVM::IsConstantOp op){return TCC_Basic;})
    .Case([](LLVM::IsFPClass op){return TCC_Basic;})
    .Case([](LLVM::LifetimeEndOp op){return TCC_Basic;})
    .Case([](LLVM::LifetimeStartOp op){return TCC_Basic;})
    .Case([](LLVM::LlrintOp op){return TCC_Basic;})
    .Case([](LLVM::LlroundOp op){return TCC_Basic;})
    .Case([](LLVM::Log10Op op){return TCC_Basic;})
    .Case([](LLVM::Log2Op op){return TCC_Basic;})
    .Case([](LLVM::LogOp op){return TCC_Basic;})
    .Case([](LLVM::LrintOp op){return TCC_Basic;})
    .Case([](LLVM::LroundOp op){return TCC_Basic;})
    .Case([](LLVM::MaskedLoadOp op){return TCC_Basic;})
    .Case([](LLVM::MaskedStoreOp op){return TCC_Basic;})
    .Case([](LLVM::MatrixColumnMajorLoadOp op){return TCC_Basic;})
    .Case([](LLVM::MatrixColumnMajorStoreOp op){return TCC_Basic;})
    .Case([](LLVM::MatrixMultiplyOp op){return TCC_Basic;})
    .Case([](LLVM::MatrixTransposeOp op){return TCC_Basic;})
    .Case([](LLVM::MaxNumOp op){return TCC_Basic;})
    .Case([](LLVM::MaximumOp op){return TCC_Basic;})
    .Case([](LLVM::MemcpyInlineOp op){return TCC_Basic;})
    .Case([](LLVM::MemcpyOp op){return TCC_Basic;})
    .Case([](LLVM::MemmoveOp op){return TCC_Basic;})
    .Case([](LLVM::MemsetOp op){return TCC_Basic;})
    .Case([](LLVM::MinNumOp op){return TCC_Basic;})
    .Case([](LLVM::MinimumOp op){return TCC_Basic;})
    .Case([](LLVM::NearbyintOp op){return TCC_Basic;})
    .Case([](LLVM::NoAliasScopeDeclOp op){return TCC_Basic;})
    .Case([](LLVM::PowIOp op){return TCC_Basic;})
    .Case([](LLVM::PowOp op){return TCC_Basic;})
    .Case([](LLVM::PtrAnnotation op){return TCC_Basic;})
    .Case([](LLVM::RintOp op){return TCC_Basic;})
    .Case([](LLVM::RoundEvenOp op){return TCC_Basic;})
    .Case([](LLVM::RoundOp op){return TCC_Basic;})
    .Case([](LLVM::SAddSat op){return TCC_Basic;})
    .Case([](LLVM::SAddWithOverflowOp op){return TCC_Basic;})
    .Case([](LLVM::SMaxOp op){return TCC_Basic;})
    .Case([](LLVM::SMinOp op){return TCC_Basic;})
    .Case([](LLVM::SMulWithOverflowOp op){return TCC_Basic;})
    .Case([](LLVM::SSACopyOp op){return TCC_Basic;})
    .Case([](LLVM::SSHLSat op){return TCC_Basic;})
    .Case([](LLVM::SSubSat op){return TCC_Basic;})
    .Case([](LLVM::SSubWithOverflowOp op){return TCC_Basic;})
    .Case([](LLVM::SinOp op){return TCC_Basic;})
    .Case([](LLVM::SqrtOp op){return TCC_Basic;})
    .Case([](LLVM::StackRestoreOp op){return TCC_Basic;})
    .Case([](LLVM::StackSaveOp op){return TCC_Basic;})
    .Case([](LLVM::StepVectorOp op){return TCC_Basic;})
    .Case([](LLVM::ThreadlocalAddressOp op){return TCC_Basic;})
    .Case([](LLVM::UAddWithOverflowOp op){return TCC_Basic;})
    .Case([](LLVM::UBSanTrap op){return TCC_Basic;})
    .Case([](LLVM::UMaxOp op){return TCC_Basic;})
    .Case([](LLVM::UMinOp op){return TCC_Basic;})
    .Case([](LLVM::UMulWithOverflowOp op){return TCC_Basic;})
    .Case([](LLVM::USHLSat op){return TCC_Basic;})
    .Case([](LLVM::USubSat op){return TCC_Basic;})
    .Case([](LLVM::USubWithOverflowOp op){return TCC_Basic;})
    .Case([](LLVM::VPAShrOp op){return TCC_Basic;})
    .Case([](LLVM::VPAddOp op){return TCC_Basic;})
    .Case([](LLVM::VPAndOp op){return TCC_Basic;})
    .Case([](LLVM::VPFAddOp op){return TCC_Basic;})
    .Case([](LLVM::VPFDivOp op){return TCC_Basic;})
    .Case([](LLVM::VPFMulAddOp op){return TCC_Basic;})
    .Case([](LLVM::VPFMulOp op){return TCC_Basic;})
    .Case([](LLVM::VPFNegOp op){return TCC_Basic;})
    .Case([](LLVM::VPFPExtOp op){return TCC_Basic;})
    .Case([](LLVM::VPFPToSIOp op){return TCC_Basic;})
    .Case([](LLVM::VPFPToUIOp op){return TCC_Basic;})
    .Case([](LLVM::VPFPTruncOp op){return TCC_Basic;})
    .Case([](LLVM::VPFRemOp op){return TCC_Basic;})
    .Case([](LLVM::VPFSubOp op){return TCC_Basic;})
    .Case([](LLVM::VPFmaOp op){return TCC_Basic;})
    .Case([](LLVM::VPIntToPtrOp op){return TCC_Basic;})
    .Case([](LLVM::VPLShrOp op){return TCC_Basic;})
    .Case([](LLVM::VPLoadOp op){return TCC_Basic;})
    .Case([](LLVM::VPMergeMinOp op){return TCC_Basic;})
    .Case([](LLVM::VPMulOp op){return TCC_Basic;})
    .Case([](LLVM::VPOrOp op){return TCC_Basic;})
    .Case([](LLVM::VPPtrToIntOp op){return TCC_Basic;})
    .Case([](LLVM::VPReduceAddOp op){return TCC_Basic;})
    .Case([](LLVM::VPReduceAndOp op){return TCC_Basic;})
    .Case([](LLVM::VPReduceFAddOp op){return TCC_Basic;})
    .Case([](LLVM::VPReduceFMaxOp op){return TCC_Basic;})
    .Case([](LLVM::VPReduceFMinOp op){return TCC_Basic;})
    .Case([](LLVM::VPReduceFMulOp op){return TCC_Basic;})
    .Case([](LLVM::VPReduceMulOp op){return TCC_Basic;})
    .Case([](LLVM::VPReduceOrOp op){return TCC_Basic;})
    .Case([](LLVM::VPReduceSMaxOp op){return TCC_Basic;})
    .Case([](LLVM::VPReduceSMinOp op){return TCC_Basic;})
    .Case([](LLVM::VPReduceUMaxOp op){return TCC_Basic;})
    .Case([](LLVM::VPReduceUMinOp op){return TCC_Basic;})
    .Case([](LLVM::VPReduceXorOp op){return TCC_Basic;})
    .Case([](LLVM::VPSDivOp op){return TCC_Basic;})
    .Case([](LLVM::VPSExtOp op){return TCC_Basic;})
    .Case([](LLVM::VPSIToFPOp op){return TCC_Basic;})
    .Case([](LLVM::VPSRemOp op){return TCC_Basic;})
    .Case([](LLVM::VPShlOp op){return TCC_Basic;})
    .Case([](LLVM::VPStoreOp op){return TCC_Basic;})
    .Case([](LLVM::VPStridedLoadOp op){return TCC_Basic;})
    .Case([](LLVM::VPStridedStoreOp op){return TCC_Basic;})
    .Case([](LLVM::VPSubOp op){return TCC_Basic;})
    .Case([](LLVM::VPTruncOp op){return TCC_Basic;})
    .Case([](LLVM::VPUDivOp op){return TCC_Basic;})
    .Case([](LLVM::VPUIToFPOp op){return TCC_Basic;})
    .Case([](LLVM::VPURemOp op){return TCC_Basic;})
    .Case([](LLVM::VPXorOp op){return TCC_Basic;})
    .Case([](LLVM::VPZExtOp op){return TCC_Basic;})
    .Case([](LLVM::VaCopyOp op){return TCC_Basic;})
    .Case([](LLVM::VaEndOp op){return TCC_Basic;})
    .Case([](LLVM::VaStartOp op){return TCC_Basic;})
    .Case([](LLVM::VarAnnotation op){return TCC_Basic;})
    .Case([](LLVM::masked_compressstore op){return TCC_Basic;})
    .Case([](LLVM::masked_expandload op){return TCC_Basic;})
    .Case([](LLVM::masked_gather op){return TCC_Basic;})
    .Case([](LLVM::masked_scatter op){return TCC_Basic;})
    .Case([](LLVM::vector_extract op){return TCC_Basic;})
    .Case([](LLVM::vector_insert op){return TCC_Basic;})
    .Case([](LLVM::vector_reduce_add op){return TCC_Basic;})
    .Case([](LLVM::vector_reduce_and op){return TCC_Basic;})
    .Case([](LLVM::vector_reduce_fadd op){return TCC_Basic;})
    .Case([](LLVM::vector_reduce_fmax op){return TCC_Basic;})
    .Case([](LLVM::vector_reduce_fmaximum op){return TCC_Basic;})
    .Case([](LLVM::vector_reduce_fmin op){return TCC_Basic;})
    .Case([](LLVM::vector_reduce_fminimum op){return TCC_Basic;})
    .Case([](LLVM::vector_reduce_mul op){return TCC_Basic;})
    .Case([](LLVM::vector_reduce_or op){return TCC_Basic;})
    .Case([](LLVM::vector_reduce_smax op){return TCC_Basic;})
    .Case([](LLVM::vector_reduce_smin op){return TCC_Basic;})
    .Case([](LLVM::vector_reduce_umax op){return TCC_Basic;})
    .Case([](LLVM::vector_reduce_umin op){return TCC_Basic;})
    .Case([](LLVM::vector_reduce_xor op){return TCC_Basic;})
    .Case([](LLVM::vscale op){return TCC_Basic;})
    ;
    timingCost += operationCost;});   
}

std::unique_ptr<Pass> mlir::createGraphPartitionPass(raw_ostream &os) {
  return std::make_unique<PartitionPass>(os);
}