# HaradiBots Architecture v3.1 — Reality-Aligned Draft

Status: draft correction to v3.0. This document does not claim production
readiness. It separates verified behavior from designed or unimplemented
behavior.

## 1. Verified baseline

- Strict versioned envelopes and progress events exist.
- Authentication is mandatory. Trusted local interfaces bootstrap an internal
  credential; the network API requires an explicit credential.
- Hardware profiling, Hugging Face metadata inspection, strategy calculation,
  lazy backend imports, mandatory teardown, and telemetry separation have
  automated coverage.
- GGUF repositories can now be resolved to a sandboxed local artifact and
  parsed for planning metadata.
- SQLite stores job metadata only. Local Redis is the standalone hot path.
- Automated suite status at this revision: 33 tests passed before the v3.1
  fail-closed validation correction.

## 2. Production blockers discovered by real Kaggle testing

### 2.1 Missing operation semantics

The v3.0 Job Envelope describes a model and strategy mode but not the requested
operation. A production contract must distinguish:

- `inspect`
- `quantize`
- `validate`
- `infer`

Without this field, inference output can be mislabeled as quantization and the
Orchestrator cannot apply operation-specific prerequisites.

### 2.2 Missing source and candidate distinction

Perplexity validation requires both an original reference model and a candidate
artifact. A single `model_source` cannot represent both. v3.1 requires:

- `source_model`: canonical original/reference model
- `candidate_artifact`: optional compiled or quantized artifact
- `target`: requested format and output location

### 2.3 Model acquisition was disconnected from execution

The inspector accepted repository IDs while llama.cpp required a local GGUF
path. The repaired lifecycle is:

1. inspect repository;
2. identify source format;
3. reject unsafe conversion;
4. select one compatible artifact or conversion input;
5. verify storage capacity;
6. acquire into the sandbox cache;
7. parse artifact-native metadata;
8. select a compatible backend;
9. register the worker;
10. execute and teardown.

### 2.4 Validation was not connected to job success

Worker-specific output dictionaries are not `ValidationResult` objects.
Artifact delivery must fail closed unless the required original-versus-candidate
validation result is produced and schema-valid. A successful subprocess alone
is not a successful quantization job.

## 3. Required v3.1 Job Envelope additions

The v3.0 fields remain, with these required additions:

| Field | Type | Purpose |
|---|---|---|
| operation | enum | inspect, quantize, validate, or infer |
| source_model | object | canonical reference repository/path and revision |
| candidate_artifact | object or null | artifact being validated or served |
| target | object or null | requested output format, quantization settings, and destination |
| validation_policy | object | required domains, thresholds, golden prompts, and fail-closed behavior |

`model_source` becomes a deprecated compatibility alias during one migration
window. Schema version must become `3.1`; v3.0 must not silently reinterpret
the new fields.

## 4. Operation-specific state paths

### Inspect

AUTHENTICATE → PROFILE → INSPECT → COMPLETE → TEARDOWN → IDLE

### Quantize

AUTHENTICATE → PROFILE → INSPECT → PLAN → ACQUIRE_SOURCE → CONVERT →
VALIDATE → PERSIST → TEARDOWN → IDLE

### Validate

AUTHENTICATE → PROFILE → ACQUIRE_REFERENCE → ACQUIRE_CANDIDATE → VALIDATE →
PERSIST → TEARDOWN → IDLE

The validate path now has concrete evaluator adapters: Transformers evaluates
canonical repositories in-process with lazy imports, while GGUF evaluation uses
the separately packaged `llama-perplexity` subprocess. Both evaluators receive
identical domain text. A validation result is persisted only after strict
schema validation; critical results fail closed.

### Infer

AUTHENTICATE → PROFILE → INSPECT → ACQUIRE_ARTIFACT → LOAD → INFER →
TEARDOWN → IDLE

Every error and cancellation path enters TEARDOWN. `complete` must include an
explicit terminal outcome and cannot imply success merely because teardown
completed.

## 5. Backend compatibility gate

Backend selection must consider all of:

- hardware profile;
- source artifact format;
- requested operation;
- requested target format;
- backend availability;
- native tokenizer availability;
- validation capability.

Hardware-only backend selection is forbidden. Examples:

- GGUF inference routes to llama.cpp.
- A SafeTensor Transformers repository cannot be sent directly to llama.cpp.
- Pre-quantized AWQ/GPTQ/EXL2 cannot be treated as an original full-precision
  source.
- A backend that cannot expose token likelihoods cannot claim perplexity
  validation without a defined external evaluator.

## 6. Success and delivery policy

A quantization job is successful only when:

1. conversion exits successfully;
2. the artifact exists and passes format inspection;
3. validation returns a strict `ValidationResult`;
4. severity policy permits delivery;
5. metadata and validation summaries are persisted;
6. teardown harvests every registered process.

Poor results remain blocked pending confirmation. Critical results are
quarantined. Missing validation is a failure, not a pass.

## 7. Deployment truth

The primary target remains a single-machine Enterprise Fat Binary:

- dependencies and llama.cpp binaries are bundled;
- Redis runs locally as a managed child process;
- SQLite stores local metadata;
- no runtime compiler or background dependency download is required.

Kaggle and Colab are development/integration adapters, not the primary
distribution. They may build llama.cpp during testing, but that behavior must
not leak into the Fat Binary.

Ray, SLURM, Kubernetes, PostgreSQL, remote Prometheus, and mTLS cluster behavior
remain optional server features. Cluster work must not begin until the
single-node production gate passes.

## 8. Production acceptance gate

The project is not production-ready until all of the following are evidenced:

- real GGUF inference on CPU and CUDA;
- a real conversion from an approved full-precision source;
- original-versus-candidate perplexity across all three built-in domains;
- golden-prompt reporting;
- poor/critical delivery controls;
- persistence and restart recovery;
- cancellation and forced-kill teardown;
- local Redis lifecycle management;
- packaged offline Fat Binary execution on a clean Windows machine;
- security, dependency, and artifact-integrity checks;
- coverage and integration tests required by Phase 14.

Until then, documentation must use “implemented,” “unit tested,” “integration
tested,” or “externally verified” precisely and must not use “production-ready.”
