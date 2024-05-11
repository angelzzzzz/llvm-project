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

void Node::calculateCost(Block *block) {}

void Node::calculateSpaceCost(Block *block) {
  spaceCost = block->getOperations().size() * SIZE_TRANSFORM_PARAMETER;
}

void Node::calculateTimingCost(Block *block, int64_t freeCycle,
                               int64_t basicCycle, int64_t expensiveCycle) {
  block->walk([&](Operation *op) {
    int64_t operationCost =
        TypeSwitch<Operation *, unsigned>(op)
            .Case([&](LLVM::AShrOp op) { return basicCycle; })
            .Case([&](LLVM::AddOp op) { return basicCycle; })
            .Case([&](LLVM::AddrSpaceCastOp op) { return basicCycle; })
            .Case([&](LLVM::AddressOfOp op) { return basicCycle; })
            .Case([&](LLVM::AllocaOp op) { return basicCycle; })
            .Case([&](LLVM::AndOp op) { return basicCycle; })
            .Case([&](LLVM::AtomicCmpXchgOp op) { return expensiveCycle; })
            .Case([&](LLVM::AtomicRMWOp op) { return expensiveCycle; })
            .Case([&](LLVM::BitcastOp op) { return basicCycle; })
            .Case([&](LLVM::BrOp op) { return basicCycle; })
            .Case([&](LLVM::CallIntrinsicOp op) { return freeCycle; })
            .Case([&](LLVM::CallOp op) { return freeCycle; })
            .Case([&](LLVM::ComdatOp op) { return freeCycle; })
            .Case([&](LLVM::ComdatSelectorOp op) { return freeCycle; })
            .Case([&](LLVM::CondBrOp op) { return basicCycle; })
            .Case([&](LLVM::ConstantOp op) { return basicCycle; })
            .Case([&](LLVM::ExtractElementOp op) { return basicCycle; })
            .Case([&](LLVM::ExtractValueOp op) { return basicCycle; })
            .Case([&](LLVM::FCmpOp op) { return basicCycle; })
            .Case([&](LLVM::FDivOp op) { return expensiveCycle; })
            .Case([&](LLVM::FMulOp op) { return basicCycle; })
            .Case([&](LLVM::FNegOp op) { return basicCycle; })
            .Case([&](LLVM::FPExtOp op) { return basicCycle; })
            .Case([&](LLVM::FPToSIOp op) { return basicCycle; })
            .Case([&](LLVM::FPToUIOp op) { return basicCycle; })
            .Case([&](LLVM::FPTruncOp op) { return basicCycle; })
            .Case([&](LLVM::FSubOp op) { return basicCycle; })
            .Case([&](LLVM::FenceOp op) { return basicCycle; })
            .Case([&](LLVM::FreezeOp op) { return freeCycle; })
            .Case([&](LLVM::GEPOp op) { return basicCycle; })
            .Case([&](LLVM::GlobalCtorsOp op) { return expensiveCycle; })
            .Case([&](LLVM::GlobalDtorsOp op) { return expensiveCycle; })
            .Case([&](LLVM::GlobalOp op) { return basicCycle; })
            .Case([&](LLVM::ICmpOp op) { return basicCycle; })
            .Case([&](LLVM::InlineAsmOp op) { return expensiveCycle; })
            .Case([&](LLVM::InsertElementOp op) { return basicCycle; })
            .Case([&](LLVM::InsertValueOp op) { return basicCycle; })
            .Case([&](LLVM::IntToPtrOp op) { return basicCycle; })
            .Case([&](LLVM::InvokeOp op) { return expensiveCycle; })
            .Case([&](LLVM::LLVMFuncOp op) { return freeCycle; })
            .Case([&](LLVM::LShrOp op) { return basicCycle; })
            .Case([&](LLVM::LandingpadOp op) { return expensiveCycle; })
            .Case([&](LLVM::LinkerOptionsOp op) { return basicCycle; })
            .Case([&](LLVM::LoadOp op) { return expensiveCycle; })
            .Case([&](LLVM::MulOp op) { return basicCycle; })
            .Case([&](LLVM::NoneTokenOp op) { return freeCycle; })
            .Case([&](LLVM::OrOp op) { return basicCycle; })
            .Case([&](LLVM::PoisonOp op) { return freeCycle; })
            .Case([&](LLVM::PtrToIntOp op) { return basicCycle; })
            .Case([&](LLVM::ResumeOp op) { return expensiveCycle; })
            .Case([&](LLVM::ReturnOp op) { return basicCycle; })
            .Case([&](LLVM::SDivOp op) { return expensiveCycle; })
            .Case([&](LLVM::SExtOp op) { return basicCycle; })
            .Case([&](LLVM::SIToFPOp op) { return basicCycle; })
            .Case([&](LLVM::SRemOp op) { return expensiveCycle; })
            .Case([&](LLVM::SelectOp op) { return basicCycle; })
            .Case([&](LLVM::ShlOp op) { return basicCycle; })
            .Case([&](LLVM::ShuffleVectorOp op) { return basicCycle; })
            .Case([&](LLVM::StoreOp op) { return expensiveCycle; })
            .Case([&](LLVM::SubOp op) { return basicCycle; })
            .Case([&](LLVM::SwitchOp op) { return basicCycle; })
            .Case([&](LLVM::TruncOp op) { return basicCycle; })
            .Case([&](LLVM::UDivOp op) { return expensiveCycle; })
            .Case([&](LLVM::UIToFPOp op) { return basicCycle; })
            .Case([&](LLVM::URemOp op) { return expensiveCycle; })
            .Case([&](LLVM::UndefOp op) { return freeCycle; })
            .Case([&](LLVM::UnreachableOp op) { return freeCycle; })
            .Case([&](LLVM::XOrOp op) { return basicCycle; })
            .Case([&](LLVM::ZExtOp op) { return basicCycle; })
            .Case([&](LLVM::ZeroOp op) { return freeCycle; })
            .Case([&](LLVM::AbsOp op) { return basicCycle; })
            .Case([&](LLVM::Annotation op) { return freeCycle; })
            .Case([&](LLVM::AssumeOp op) { return basicCycle; })
            .Case([&](LLVM::BitReverseOp op) { return expensiveCycle; })
            .Case([&](LLVM::ByteSwapOp op) { return basicCycle; })
            .Case(
                [&](LLVM::ConstrainedFPTruncIntr op) { return expensiveCycle; })
            .Case([&](LLVM::CopySignOp op) { return basicCycle; })
            .Case([&](LLVM::CoroAlignOp op) { return basicCycle; })
            .Case([&](LLVM::CoroBeginOp op) { return expensiveCycle; })
            .Case([&](LLVM::CoroEndOp op) { return basicCycle; })
            .Case([&](LLVM::CoroFreeOp op) { return basicCycle; })
            .Case([&](LLVM::CoroIdOp op) { return basicCycle; })
            .Case([&](LLVM::CoroPromiseOp op) { return basicCycle; })
            .Case([&](LLVM::CoroResumeOp op) { return expensiveCycle; })
            .Case([&](LLVM::CoroSaveOp op) { return basicCycle; })
            .Case([&](LLVM::CoroSizeOp op) { return basicCycle; })
            .Case([&](LLVM::CoroSuspendOp op) { return expensiveCycle; })
            .Case([&](LLVM::CosOp op) { return basicCycle; })
            .Case([&](LLVM::CountLeadingZerosOp op) { return basicCycle; })
            .Case([&](LLVM::CountTrailingZerosOp op) { return basicCycle; })
            .Case([&](LLVM::CtPopOp op) { return basicCycle; })
            .Case([&](LLVM::DbgDeclareOp op) { return basicCycle; })
            .Case([&](LLVM::DbgLabelOp op) { return basicCycle; })
            .Case([&](LLVM::DbgValueOp op) { return basicCycle; })
            .Case([&](LLVM::EhTypeidForOp op) { return basicCycle; })
            .Case([&](LLVM::Exp2Op op) { return expensiveCycle; })
            .Case([&](LLVM::ExpOp op) { return expensiveCycle; })
            .Case([&](LLVM::ExpectOp op) { return basicCycle; })
            .Case([&](LLVM::ExpectWithProbabilityOp op) { return basicCycle; })
            .Case([&](LLVM::FAbsOp op) { return basicCycle; })
            .Case([&](LLVM::FCeilOp op) { return basicCycle; })
            .Case([&](LLVM::FFloorOp op) { return basicCycle; })
            .Case([&](LLVM::FMAOp op) { return basicCycle; })
            .Case([&](LLVM::FMulAddOp op) { return basicCycle; })
            .Case([&](LLVM::FTruncOp op) { return basicCycle; })
            .Case([&](LLVM::FshlOp op) { return basicCycle; })
            .Case([&](LLVM::FshrOp op) { return basicCycle; })
            .Case([&](LLVM::GetActiveLaneMaskOp op) { return basicCycle; })
            .Case([&](LLVM::InvariantEndOp op) { return basicCycle; })
            .Case([&](LLVM::InvariantStartOp op) { return basicCycle; })
            .Case([&](LLVM::IsConstantOp op) { return basicCycle; })
            .Case([&](LLVM::IsFPClass op) { return basicCycle; })
            .Case([&](LLVM::LifetimeEndOp op) { return basicCycle; })
            .Case([&](LLVM::LifetimeStartOp op) { return basicCycle; })
            .Case([&](LLVM::LlrintOp op) { return expensiveCycle; })
            .Case([&](LLVM::LlroundOp op) { return expensiveCycle; })
            .Case([&](LLVM::Log10Op op) { return expensiveCycle; })
            .Case([&](LLVM::Log2Op op) { return expensiveCycle; })
            .Case([&](LLVM::LogOp op) { return expensiveCycle; })
            .Case([&](LLVM::LrintOp op) { return expensiveCycle; })
            .Case([&](LLVM::LroundOp op) { return expensiveCycle; })
            .Case([&](LLVM::MaskedLoadOp op) { return expensiveCycle; })
            .Case([&](LLVM::MaskedStoreOp op) { return expensiveCycle; })
            .Case([&](LLVM::MatrixColumnMajorLoadOp op) {
              return expensiveCycle;
            })
            .Case([&](LLVM::MatrixColumnMajorStoreOp op) {
              return expensiveCycle;
            })
            .Case([&](LLVM::MatrixMultiplyOp op) { return expensiveCycle; })
            .Case([&](LLVM::MatrixTransposeOp op) { return basicCycle; })
            .Case([&](LLVM::MaxNumOp op) { return basicCycle; })
            .Case([&](LLVM::MaximumOp op) { return basicCycle; })
            .Case([&](LLVM::MemcpyInlineOp op) { return basicCycle; })
            .Case([&](LLVM::MemcpyOp op) { return expensiveCycle; })
            .Case([&](LLVM::MemmoveOp op) { return expensiveCycle; })
            .Case([&](LLVM::MemsetOp op) { return basicCycle; })
            .Case([&](LLVM::MinNumOp op) { return basicCycle; })
            .Case([&](LLVM::MinimumOp op) { return basicCycle; })
            .Case([&](LLVM::NearbyintOp op) { return basicCycle; })
            .Case([&](LLVM::NoAliasScopeDeclOp op) { return basicCycle; })
            .Case([&](LLVM::PowIOp op) { return expensiveCycle; })
            .Case([&](LLVM::PowOp op) { return expensiveCycle; })
            .Case([&](LLVM::PtrAnnotation op) { return basicCycle; })
            .Case([&](LLVM::RintOp op) { return expensiveCycle; })
            .Case([&](LLVM::RoundEvenOp op) { return expensiveCycle; })
            .Case([&](LLVM::RoundOp op) { return expensiveCycle; })
            .Case([&](LLVM::SAddSat op) { return basicCycle; })
            .Case([&](LLVM::SAddWithOverflowOp op) { return basicCycle; })
            .Case([&](LLVM::SMaxOp op) { return basicCycle; })
            .Case([&](LLVM::SMinOp op) { return basicCycle; })
            .Case([&](LLVM::SMulWithOverflowOp op) { return expensiveCycle; })
            .Case([&](LLVM::SSACopyOp op) { return basicCycle; })
            .Case([&](LLVM::SSHLSat op) { return basicCycle; })
            .Case([&](LLVM::SSubSat op) { return basicCycle; })
            .Case([&](LLVM::SSubWithOverflowOp op) { return expensiveCycle; })
            .Case([&](LLVM::SinOp op) { return expensiveCycle; })
            .Case([&](LLVM::SqrtOp op) { return expensiveCycle; })
            .Case([&](LLVM::StackRestoreOp op) { return basicCycle; })
            .Case([&](LLVM::StackSaveOp op) { return basicCycle; })
            .Case([&](LLVM::StepVectorOp op) { return basicCycle; })
            .Case([&](LLVM::ThreadlocalAddressOp op) { return expensiveCycle; })
            .Case([&](LLVM::UAddWithOverflowOp op) { return expensiveCycle; })
            .Case([&](LLVM::UBSanTrap op) { return expensiveCycle; })
            .Case([&](LLVM::UMaxOp op) { return basicCycle; })
            .Case([&](LLVM::UMinOp op) { return basicCycle; })
            .Case([&](LLVM::UMulWithOverflowOp op) { return expensiveCycle; })
            .Case([&](LLVM::USHLSat op) { return basicCycle; })
            .Case([&](LLVM::USubSat op) { return basicCycle; })
            .Case([&](LLVM::USubWithOverflowOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPAShrOp op) { return basicCycle; })
            .Case([&](LLVM::VPAddOp op) { return basicCycle; })
            .Case([&](LLVM::VPAndOp op) { return basicCycle; })
            .Case([&](LLVM::VPFAddOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPFDivOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPFMulAddOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPFMulOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPFNegOp op) { return basicCycle; })
            .Case([&](LLVM::VPFPExtOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPFPToSIOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPFPToUIOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPFPTruncOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPFRemOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPFSubOp op) { return basicCycle; })
            .Case([&](LLVM::VPFmaOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPIntToPtrOp op) { return basicCycle; })
            .Case([&](LLVM::VPLShrOp op) { return basicCycle; })
            .Case([&](LLVM::VPLoadOp op) { return basicCycle; })
            .Case([&](LLVM::VPMergeMinOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPMulOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPOrOp op) { return basicCycle; })
            .Case([&](LLVM::VPPtrToIntOp op) { return basicCycle; })
            .Case([&](LLVM::VPReduceAddOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPReduceAndOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPReduceFAddOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPReduceFMaxOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPReduceFMinOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPReduceFMulOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPReduceMulOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPReduceOrOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPReduceSMaxOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPReduceSMinOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPReduceUMaxOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPReduceUMinOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPReduceXorOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPSDivOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPSExtOp op) { return basicCycle; })
            .Case([&](LLVM::VPSIToFPOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPSRemOp op) { return basicCycle; })
            .Case([&](LLVM::VPShlOp op) { return basicCycle; })
            .Case([&](LLVM::VPStoreOp op) { return basicCycle; })
            .Case([&](LLVM::VPStridedLoadOp op) { return basicCycle; })
            .Case([&](LLVM::VPStridedStoreOp op) { return basicCycle; })
            .Case([&](LLVM::VPSubOp op) { return basicCycle; })
            .Case([&](LLVM::VPTruncOp op) { return basicCycle; })
            .Case([&](LLVM::VPUDivOp op) { return basicCycle; })
            .Case([&](LLVM::VPUIToFPOp op) { return basicCycle; })
            .Case([&](LLVM::VPURemOp op) { return expensiveCycle; })
            .Case([&](LLVM::VPXorOp op) { return basicCycle; })
            .Case([&](LLVM::VPZExtOp op) { return basicCycle; })
            .Case([&](LLVM::VaCopyOp op) { return basicCycle; })
            .Case([&](LLVM::VaEndOp op) { return basicCycle; })
            .Case([&](LLVM::VaStartOp op) { return basicCycle; })
            .Case([&](LLVM::VarAnnotation op) { return basicCycle; })
            .Case([&](LLVM::masked_compressstore op) { return expensiveCycle; })
            .Case([&](LLVM::masked_expandload op) { return expensiveCycle; })
            .Case([&](LLVM::masked_gather op) { return expensiveCycle; })
            .Case([&](LLVM::masked_scatter op) { return expensiveCycle; })
            .Case([&](LLVM::vector_extract op) { return basicCycle; })
            .Case([&](LLVM::vector_insert op) { return basicCycle; })
            .Case([&](LLVM::vector_reduce_add op) { return expensiveCycle; })
            .Case([&](LLVM::vector_reduce_and op) { return expensiveCycle; })
            .Case([&](LLVM::vector_reduce_fadd op) { return expensiveCycle; })
            .Case([&](LLVM::vector_reduce_fmax op) { return expensiveCycle; })
            .Case(
                [&](LLVM::vector_reduce_fmaximum op) { return expensiveCycle; })
            .Case([&](LLVM::vector_reduce_fmin op) { return expensiveCycle; })
            .Case(
                [&](LLVM::vector_reduce_fminimum op) { return expensiveCycle; })
            .Case([&](LLVM::vector_reduce_mul op) { return expensiveCycle; })
            .Case([&](LLVM::vector_reduce_or op) { return expensiveCycle; })
            .Case([&](LLVM::vector_reduce_smax op) { return expensiveCycle; })
            .Case([&](LLVM::vector_reduce_smin op) { return expensiveCycle; })
            .Case([&](LLVM::vector_reduce_umax op) { return expensiveCycle; })
            .Case([&](LLVM::vector_reduce_umin op) { return expensiveCycle; })
            .Case([&](LLVM::vector_reduce_xor op) { return expensiveCycle; });
    timingCost += operationCost;
  });
}

std::unique_ptr<Pass> mlir::createGraphPartitionPass(raw_ostream &os) {
  return std::make_unique<PartitionPass>(os);
}