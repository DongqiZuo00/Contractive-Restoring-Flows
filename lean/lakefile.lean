import Lake
open Lake DSL

package crf_verify where
  leanOptions := #[⟨`autoImplicit, false⟩]

@[default_target]
lean_lib CRF where
  srcDir := "."
  roots := #[`CRF_Complete]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"
