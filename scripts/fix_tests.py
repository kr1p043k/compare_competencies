"""Fix is_compatible mocks across all test files."""
import re

files = {
    "tests/pipeline/test_skill_extractor_extended.py": [
        (
            'return_value=Ok(MagicMock(is_compatible=MagicMock(return_value=True)))',
            'return_value=Ok(MagicMock(is_compatible=MagicMock(return_value=Ok(True))))',
        ),
        (
            'incompatible.is_compatible.return_value = False',
            'from src import Ok; incompatible.is_compatible.return_value = Ok(False)',
        ),
    ],
    "tests/predictors/test_ltr_recommendation_engine.py": [
        ("mock_instance.is_compatible.return_value = False", "mock_instance.is_compatible.return_value = Ok(False)"),
        ("mock_inst.is_compatible.return_value = True;", "mock_inst.is_compatible.return_value = Ok(True);"),
        ("mock_inst.is_compatible.return_value = True\n", "mock_inst.is_compatible.return_value = Ok(True)\n"),
    ],
}

for fpath, replacements in files.items():
    full = __import__("os").path.join(__import__("os").path.dirname(__file__), "..", fpath)
    with open(full, "r", encoding="utf-8") as f:
        content = f.read()
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"  {fpath}: replaced")
        else:
            # try partial match
            pass
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    # Check and add Ok import if needed
    if "Ok(True)" in content or "Ok(False)" in content:
        lines = content.split("\n")
        has_ok_import = any("from src import Ok" in l or "from src import Ok," in l for l in lines)
        if not has_ok_import:
            for i, line in enumerate(lines):
                if line.strip().startswith("from src import"):
                    if "Ok" not in line:
                        lines[i] = line.replace("from src import", "from src import Ok, ")
                    break
            with open(full, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"  {fpath}: added Ok import")
    print(f"  {fpath}: done")
