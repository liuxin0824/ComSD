# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

from base.learners.skill_discovery.base import BaseSkillDiscoveryLearner


class BaseForwardMILearner(BaseSkillDiscoveryLearner):
    AGENT_TYPE = 'ForwardMI'

    def __init__(self, skill_n, **kwargs):
        self.skill_n = int(skill_n)

        # At least trigger the default usage (if default im_params=None is used the im module will be skipped)
        if 'im_params' not in kwargs:
            kwargs['im_params'] = {}
        super().__init__(**kwargs)
        self.im_type = 'forward_mi'
