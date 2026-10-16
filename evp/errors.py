#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Typed errors with actionable hints. CLI prints `hint` and exits non-zero."""

from __future__ import annotations


class EvpError(Exception):
    """Base error. `hint` tells the user what to do next."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\n  hint: {self.hint}"
        return self.message


class ConfigError(EvpError):
    pass


class DataError(EvpError):
    pass


class ExportError(EvpError):
    pass


class ConvertError(EvpError):
    pass


class UnsupportedOpError(ConvertError):
    """Risky / unsupported operators left after the rules; `ops` lists them for the CLI and tests."""

    def __init__(self, ops: list[str], hint: str | None = None):
        self.ops = sorted(set(ops))
        super().__init__(
            f"unsupported operators for ncnn: {', '.join(self.ops)}",
            hint
            or "add a cut point before these ops (deploy.onnx.cut_outputs) or enable a "
            "replacement rule; `evp inspect-onnx <model>` suggests a cut (docs/DEPLOY_TOOLCHAIN.md section 2)",
        )


class QuantError(EvpError):
    pass


class VerifyError(EvpError):
    pass


class ToolNotFoundError(EvpError):
    pass
