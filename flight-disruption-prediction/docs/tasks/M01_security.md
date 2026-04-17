# Milestone 1 — Security & Config Hygiene

> **Status**: `[x]` Complete
> **Priority**: 🔴 HIGH — Leaked credentials, broken imports
> **Depends on**: Nothing (start immediately)

---

## `[x]` TASK-1: Remove leaked credentials & fix .gitignore

**Files**: `credentials.json`, `.gitignore`, `docs/SECURITY.md` [NEW]
**What**: Delete `credentials.json`, add it to `.gitignore`, fix the null-byte corruption on the last 2 lines of `.gitignore`, and create a `docs/SECURITY.md` with a credential rotation checklist.

**Steps**:
1. Delete `credentials.json` from repo
2. Open `.gitignore`, remove trailing null-byte lines (`\x00`), append `credentials.json` and `.env`
3. Create `docs/SECURITY.md` documenting: what was exposed, rotation steps for OpenSky API keys, and a checklist for future secret handling

**Acceptance Criteria**:
- [ ] `credentials.json` does not exist on disk
- [ ] `python -c "open('.gitignore','rb').read()"` contains zero `\x00` bytes
- [ ] `.gitignore` contains entries for `credentials.json`, `.env`, and `*.pyc`
- [ ] `docs/SECURITY.md` exists and contains ≥3 rotation checklist items

---

## `[x]` TASK-2: Fix broken imports in main.py

**Files**: `main.py`
**What**: Lines 47 and 51 call `validate_environment()` and `set_seed()` without importing them. Add the missing imports from their existing modules.

**Steps**:
1. Add `from src.config_validator import validate_environment` to imports
2. Add `from src.utils import set_seed` to the existing `src.utils` import line
3. Verify existing call signatures match: `validate_environment(env_path)` and `set_seed(config.get("seed", 42))`

**Acceptance Criteria**:
- [ ] `python -c "import ast; ast.parse(open('main.py').read()); print('OK')"` prints `OK` (no syntax/import resolution errors in AST)
- [ ] `python main.py --help` (or a dry-run) does not raise `NameError`
- [ ] No other behavioural changes introduced
