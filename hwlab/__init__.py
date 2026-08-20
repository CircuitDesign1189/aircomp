# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""hwlab -- real-RF transmission layer for the AirComp semantic pipeline.

This package is deliberately isolated from `airComp/`: nothing here is imported
by the simulation-only experiments, and `airComp/` is never modified. The
hardware pipeline plugs in by *subclassing* SemanticAgent and replacing its
`channel` attribute (see `hwlab.agent`), so the published simulation results
stay reproducible even while this package is under development.

Layering:
    dsp/      pure-numpy signal processing (no hardware, fully unit-tested)
    radio/    SDRBackend implementations (LoopbackBackend needs no hardware)
    channel/  SDRAnalogChannel -- a drop-in for AnalogAWGNChannel
    agent.py  HardwareSemanticAgent
"""

__all__ = ["dsp", "radio", "channel"]
