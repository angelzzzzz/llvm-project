#ifndef MLIR_TRANSFORMS_GRAPHPARTITION_H_
#define MLIR_TRANSFORMS_GRAPHPARTITION_H_

#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class Pass;

#define GEN_PASS_DECL_GRAPHPARTITION
#include "mlir/Transforms/Passes.h.inc"

/// Creates a pass to part op graphs.
std::unique_ptr<Pass> createGraphPartitionPass(raw_ostream &os = llvm::errs());

} // namespace mlir

#endif // MLIR_TRANSFORMS_GRAPHPARTITION_H_
