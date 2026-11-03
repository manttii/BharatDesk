"""Principle 2: money is integer paise, weight is integer milligrams. Never float.

Three independent guards:
  1. storage: every *_paise / *_mg column is an integer type, and no Float/Numeric exists;
  2. schemas: money/weight/bps fields in pydantic models are strict ints;
  3. source: the money-touching modules contain no float literal, no float()/round()/Decimal()
     call and no `/` (true division yields a float even for ints).
"""

from __future__ import annotations

import ast
import inspect
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic.types import Strict
from sqlalchemy import BigInteger, Float, Integer, Numeric

from app.models import Base
from app.schemas import extraction as extraction_schemas
from app.schemas import policy as policy_schemas
from app.schemas import slip as slip_schemas

MONEY_WEIGHT_SUFFIXES = ("_paise", "_mg", "_bps")
APP = Path(__file__).resolve().parent.parent / "app"

# Modules that touch money or weight. Extend this list whenever a new one is added.
MONEY_MODULES = [
    "units.py",
    "pipeline/numerals.py",
    "pipeline/policy_engine.py",
    "pipeline/commit.py",
    "schemas/policy.py",
    "services/ledger.py",
    "services/approvals.py",
    "services/corrections.py",
    "services/sheets_sync.py",
    "services/dossier.py",
    "pipeline/slip.py",
    "pipeline/amounts.py",
    "pipeline/expr.py",
]


def _all_columns() -> list[tuple[str, str, object]]:
    return [(t.name, c.name, c.type) for t in Base.metadata.tables.values() for c in t.columns]


def test_money_and_weight_columns_exist_and_are_integers() -> None:
    cols = [c for c in _all_columns() if c[1].endswith(MONEY_WEIGHT_SUFFIXES)]
    assert cols, "no *_paise/*_mg columns found: this test would be vacuous"
    bad = [f"{t}.{c}" for t, c, ty in cols if not isinstance(ty, Integer | BigInteger)]
    assert bad == []


def test_no_float_or_decimal_column_anywhere() -> None:
    bad = [f"{t}.{c}" for t, c, ty in _all_columns() if isinstance(ty, Float | Numeric)]
    assert bad == [], "float/decimal columns are forbidden; use integer paise / milligrams"


# --- schemas ---------------------------------------------------------------------------


def _is_strict(tp: object, field: FieldInfo | None = None) -> bool:
    if field is not None and any(isinstance(m, Strict) and m.strict for m in field.metadata):
        return True
    if typing.get_origin(tp) is typing.Annotated:
        for meta in typing.get_args(tp)[1:]:
            inner = getattr(meta, "metadata", ())
            if any(isinstance(m, Strict) and m.strict for m in (meta, *inner)):
                return True
    return any(_is_strict(a) for a in typing.get_args(tp))


def _models() -> list[type[BaseModel]]:
    return [
        m
        for module in (policy_schemas, extraction_schemas, slip_schemas)
        for _, m in inspect.getmembers(module, inspect.isclass)
        if issubclass(m, BaseModel)
        and m.__module__ == module.__name__
        # LLMConfidence holds 0..1 confidence SCORES that reuse the field names
        # (weight_mg, value_paise, ...). They are not amounts, so they are exempt.
        and m.__name__ != "LLMConfidence"
    ]


def test_money_weight_and_bps_schema_fields_are_strict_ints() -> None:
    checked = 0
    for model in _models():
        for name, info in model.model_fields.items():
            if name.endswith(MONEY_WEIGHT_SUFFIXES):
                checked += 1
                assert _is_strict(info.annotation, info), f"{model.__name__}.{name} is not strict"
    assert checked >= 17, "too few money fields found; the introspection is probably broken"


# --- source ----------------------------------------------------------------------------


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            found.append(f"{path.name}:{line} float literal {node.value!r}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("float", "round", "Decimal")
        ):
            # Only CALLS: a `float` type annotation on a 0-1 confidence score is harmless.
            found.append(f"{path.name}:{line} calls {node.func.id}()")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            found.append(f"{path.name}:{line} true division '/'")
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Div):
            found.append(f"{path.name}:{line} true division '/='")
    return found


@pytest.mark.parametrize("module", MONEY_MODULES)
def test_money_modules_contain_no_float_arithmetic(module: str) -> None:
    path = APP / module
    assert path.is_file(), f"{module} listed in MONEY_MODULES but missing"
    assert _violations(path) == []


def test_the_scanner_itself_catches_floats(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("x = 0.5\ny = float(3)\nz = 7 / 2\nw = round(z)\n")
    assert len(_violations(bad)) == 4
