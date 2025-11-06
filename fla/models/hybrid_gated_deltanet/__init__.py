# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © 2023-2025, Songlin Yang, Yu Zhang
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.hybrid_gated_deltanet.configuration_hybrid_gated_deltanet import \
    HybridGatedDeltaNetConfig
from fla.models.hybrid_gated_deltanet.modeling_hybrid_gated_deltanet import (
    HybridGatedDeltaNetForCausalLM, HybridGatedDeltaNetModel)

AutoConfig.register(HybridGatedDeltaNetConfig.model_type, HybridGatedDeltaNetConfig)
AutoModel.register(HybridGatedDeltaNetConfig, HybridGatedDeltaNetModel)
AutoModelForCausalLM.register(HybridGatedDeltaNetConfig, HybridGatedDeltaNetForCausalLM)

__all__ = ['HybridGatedDeltaNetConfig', 'HybridGatedDeltaNetForCausalLM', 'HybridGatedDeltaNetModel']
