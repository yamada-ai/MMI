import os
import sys

import scipy
import re
import json
import random

class Generator:

    def __init__(self, bot_name, path) -> None:
        self.bot_name = bot_name
        self.phase = 0
        self.phase_rules = [
            self._phase_0,
            self._phase_1,
            self._phase_2,
            self._phase_3
        ]
        self._install_rule(path)

        self._update_phase(self.phase)
        
    def _update_phase(self, phase):
        self.current_phase = self.phase_rules[phase]
        self.corrent_rule = self.rule_base[phase]["rule"]
    
    def _install_rule(self, path):
        with open(path, "r") as f:
            rule_base_all = json.load(f)
        self.rule_base = rule_base_all[self.bot_name]

    def reply(self, context):
        self._update_phase(self.phase)
        rep, next_phase = self.current_phase(context)
        self._update_phase(next_phase)
        self.phase = next_phase
        print("next : {0}, current_phase: {1}".format(next_phase, self.current_phase.__name__))
        return rep
    
    def _phase_0(self, context):
        usr = context[-1]
        # no_list = set("ダメ だめ 駄目 待て 待って ... 何時 嫌 いや は？".split())
        # yes_list = set("どうぞ はい なんでし なんですか いいよ 良い ".split())

        no_list = "ダメ だめ 駄目 待て 待って ... 何時だと？ 嫌 いや は？　やだ".split()
        double_no_list = "でもない ではない 嘘 というわけでは というわけでも ということではない".split()
        yes_list = "どうぞ はい なんでし なんですか いいよ 良い ".split()

        usr_act = "yes"
        for no in no_list:
            if no in usr:
                usr_act = "no"
                # 二重否定は死ね
                for dou in double_no_list:
                    if dou in usr:
                        usr_act = "yes"
                        break
                break
        
        for rule in self.corrent_rule:
            if rule["usr_act"] == usr_act:
                if isinstance(rule["reply"], list):
                    return random.choice(rule["reply"]), rule["next"]
                else:
                    return rule["reply"], rule["next"]
    
    def _phase_1(self, context):
        usr = context[-1]

        why_list = "なんで？ 何故 なぜ 同期だけ 同期と 俺 私".split()
        who_list = "誰 同期って".split()
        no_list = "ダメ だめ 駄目 待て 待って ... 何時 嫌 いや は？ 気が乗ら".split()

        usr_act = "else"
        acts = [why_list, who_list, no_list]
        acts_str = ["why", "who", "no"]
        for act, act_s in zip(acts, acts_str):
            for word in act:
                if word in usr:
                    usr_act = act_s
        
        for rule in self.corrent_rule:
            if rule["usr_act"] == usr_act:
                if isinstance(rule["reply"], list):
                    return random.choice(rule["reply"]), rule["next"]
                else:
                    return rule["reply"], rule["next"]
    
    def _phase_2(self, context):
        usr = context[-1]

        # yes_list = "そうだ ".split()
        usr_act = "yes"
        no_list = "実は いや でも はずだ".split()
        for no in no_list:
            if no in usr:
                usr_act = "no"
                # 二重否定は死ね
                break
        for rule in self.corrent_rule:
            if rule["usr_act"] == usr_act:
                if isinstance(rule["reply"], list):
                    return random.choice(rule["reply"]), rule["next"]
                else:
                    return rule["reply"], rule["next"]


    def _phase_3(self, context):
        usr = context[-1]


