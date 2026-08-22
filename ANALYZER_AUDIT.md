# Bank Statement Analyzer Audit

Date: 2026-08-22

## Scope

This audit followed bank-statement data from PDF extraction through row
selection, direction inference, categorization, document merge, internal
transfer detection, and reporting. The separate everyday-problem solver was
not changed except that its live API tests now require explicit opt-in.

## Confirmed root causes

1. Direction was inferred from a sign or keyword and otherwise defaulted to a
   debit. That fails for unsigned debit/credit columns.
2. Summary filtering was word-based and allowed dated opening/ending-balance
   rows to look like transactions.
3. Regular PDF text extraction collapsed debit and credit column positions.
4. A post-parser income rule could override a debit merely because its
   description contained an income-like word.
5. The LLM received `is_debit` after the parser had already decided it, so a
   model change could not fix the primary direction errors.
6. Transfer matching depended on already-correct directions and a narrow set
   of description markers. Uploaded documents were also assigned different
   default owners, preventing ownership-aware matching.
7. LLM category output was validated structurally but a category outside the
   allowed list could still reach the transaction before this audit.

## Implemented controls

- PDF text extraction preserves horizontal layout.
- Headers, opening/closing balances, totals, subtotals, and summaries receive a
  `row_type` and are excluded before transaction parsing.
- Direction evidence is retained (`amount_column`, `explicit_sign`,
  `section_header`, `line_heuristic`, or `running_balance`).
- Bank-account direction is reconciled against running-balance changes.
- Unresolved directions fail closed: the row remains visible but contributes
  to neither income nor spending until reviewed.
- Chase, Wells Fargo, and generic statement profiles are detected from the
  original filename and document header rather than merchant descriptions.
- Original uploaded filenames are preserved through Streamlit temporary files.
- The transaction editor can correct `EGRESO`/`INGRESO` as well as category and
  transfer treatment.
- Transfer pairs require compatible ownership, currency, opposite direction,
  amount, date, and textual/reference evidence. Matched rows share a
  `transfer_pair_id`.
- Manual transfer choices are session controls and are not saved as merchant
  learning rules.
- LLM categorization preserves original transaction order, excludes unresolved
  directions, uses a strict JSON schema, and rejects categories outside the
  allowlist. The model remains `gpt-4o-mini` pending a measured model eval.
- Streamlit diagnostics expose reconciliation results and redacted samples of
  excluded rows.

## Verification

- Wells Fargo fixture: 26 real movements after excluding the beginning and
  ending balances; the unsigned `$3,200.00` direct deposit is a credit.
- Wells Fargo reconciliation difference: `$0.00`.
- Chase reconciliation difference: `$0.00`.
- Consolidated deterministic smoke test: 74 movements, 70 debits, 4 credits,
  with no LLM calls.
- Test suite: 55 passed, 9 live-API tests skipped by default.
- Python compilation: passed.
- Streamlit startup and `/_stcore/health`: passed.

## Remaining boundaries

- Image-only/scanned PDFs still require OCR before this parser can read them.
- New layouts without signs, recognizable columns, sections, or running
  balances will be marked for review rather than guessed.
- Credit-card statements need dedicated reconciliation fixtures because their
  accounting direction is not identical to a checking account.
- A transfer can only be paired automatically when both account sides are
  uploaded, unless the description explicitly identifies an own-account move.
- Representative anonymized failing statements should be added as regression
  fixtures before claiming support for their exact bank/layout versions.

## Repository note

The workspace's `.git` file points to a missing linked-worktree location, so
`git status` is unavailable. No destructive Git repair was attempted during
this audit.
