export const meta = {
  name: 'review-written-function',
  description: 'Adversarially review a newly written expdpy function across math, edge cases, and conventions',
  phases: [
    { title: 'Review' },
    { title: 'Verify' },
  ],
}

// Parameterized via the Workflow `args`:
//   args.files : string[]  - paths to review (e.g. ["src/expdpy/<fn>.py", "tests/test_<fn>.py"])
//   args.spec  : string    - one paragraph: the estimand, assumptions, and conventions to enforce
// Run from the skill's Phase 5:
//   Workflow({ scriptPath: ".claude/skills/write-function/scripts/review_workflow.js",
//              args: { files: [...], spec: "..." } })
const FILES = (args && Array.isArray(args.files) && args.files.length ? args.files : [
  'the new function module and its test file (find them under src/expdpy/ and tests/)',
]).join(', ')
const SPEC = (args && args.spec) ? args.spec : (
  'A newly added expdpy analysis function. Enforce: module-prefixed public name; frozen ' +
  'Interpretable result dataclass exposing .df plus .fig/.gt; estimation via pyfixest/' +
  'linearmodels (not hand-rolled); defensive input validation; NumPy docstring stating the ' +
  'math; association-only .interpret() (no "causes"/"effect of"); a pytest suite with a ' +
  'mathematical-validity test against a known baseline plus edge cases.'
)

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          title: { type: 'string' },
          file: { type: 'string' },
          line_hint: { type: 'string' },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          category: { type: 'string', enum: ['correctness', 'edge_case', 'convention', 'docs', 'reuse'] },
          explanation: { type: 'string' },
          suggested_fix: { type: 'string' },
        },
        required: ['title', 'file', 'severity', 'category', 'explanation', 'suggested_fix'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    is_real: { type: 'boolean' },
    reasoning: { type: 'string' },
    severity: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
  required: ['is_real', 'reasoning', 'severity'],
}

const DIMENSIONS = [
  {
    key: 'math',
    prompt: `You are reviewing the NUMERICAL CORRECTNESS of a newly written expdpy function.
Read these files in full: ${FILES}.
Context / spec:
${SPEC}
Scrutinize every formula and computation: the estimand and its closed form, sign/domain guards
(division by zero, log of non-positive, NaN handling), that the reported coefficient/SE come
from the right model, that any residualization/partialling-out is correct, and that the test
suite's "known baseline" actually pins the truth (not a tautology). Flag ONLY genuine
numerical or logic errors, or deviations from the stated math. Return an empty findings array
if correct. No style nits.`,
  },
  {
    key: 'edge',
    prompt: `You are reviewing EDGE-CASE ROBUSTNESS of a newly written expdpy function.
Read these files in full: ${FILES}.
Context / spec:
${SPEC}
Scrutinize: missing data / dropna logic, unbalanced panels, duplicate keys, zero-variance or
collinear regressors, too-few-observations, non-numeric or wrong-dtype inputs, NaN propagation
into outputs/tables, and whether each failure raises a clear, correctly-typed error rather than
crashing cryptically or returning silently-wrong numbers. Confirm the tests actually exercise
these. Flag real crash risks or silently-wrong results. Empty array if robust. No style nits.`,
  },
  {
    key: 'integration',
    prompt: `You are reviewing CONVENTION & INTEGRATION consistency of a newly written expdpy
function. Read these files in full: ${FILES}. Also check the wiring it depends on:
src/expdpy/_types.py, src/expdpy/__init__.py, and (if a concept explainer was added)
src/expdpy/pedagogy/_interpret.py and src/expdpy/pedagogy/_text/.
Context / spec:
${SPEC}
Enforce the project rules (see .claude/skills/write-function/references/conventions.md and
CLAUDE.md): module-prefixed public name exported in __all__; frozen Interpretable result
dataclass exposing .df plus .fig/.gt; .interpret() is association-only and ends with the shared
note (no "causes"/"effect of"); estimation goes through pyfixest/linearmodels; existing helpers
are reused rather than duplicated; full NumPy docstrings; ASCII-only source (no Unicode minus or
Greek letters in code strings). Flag missing exports, duplicated logic, docstring/claim
inaccuracies, or convention violations. Empty array if consistent.`,
  },
]

const reviewed = await pipeline(
  DIMENSIONS,
  (d) => agent(d.prompt, { label: `review:${d.key}`, phase: 'Review', schema: FINDINGS_SCHEMA }),
  (res, d) =>
    parallel(
      (res?.findings ?? []).map((f) => () =>
        agent(
          `Adversarially verify this code-review finding about a newly written expdpy function. ` +
          `Read the cited file(s) yourself and decide whether it is a REAL defect that should be ` +
          `fixed. Be skeptical: default is_real=false if the code is actually correct, the concern ` +
          `is already handled elsewhere, or it is a subjective style preference.\n` +
          `Context / spec:\n${SPEC}\n\nFINDING:\n${JSON.stringify(f, null, 2)}`,
          { label: `verify:${d.key}:${(f.title || '').slice(0, 30)}`, phase: 'Verify', schema: VERDICT_SCHEMA },
        ).then((v) => ({ finding: f, verdict: v })),
      ),
    ),
)

const all = reviewed.flat().filter(Boolean)
const confirmed = all.filter((x) => x.verdict && x.verdict.is_real)
const rejected = all.filter((x) => x.verdict && !x.verdict.is_real)

log(`Findings raised: ${all.length} - confirmed: ${confirmed.length} - rejected: ${rejected.length}`)

return {
  confirmed: confirmed.map((x) => ({
    title: x.finding.title,
    file: x.finding.file,
    line_hint: x.finding.line_hint,
    severity: x.verdict.severity || x.finding.severity,
    category: x.finding.category,
    explanation: x.finding.explanation,
    suggested_fix: x.finding.suggested_fix,
    verifier_reasoning: x.verdict.reasoning,
  })),
  rejected: rejected.map((x) => ({ title: x.finding.title, why_rejected: x.verdict.reasoning })),
}
