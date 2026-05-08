from __future__ import annotations

import math

EPS = 1e-6


class TripleLayerFeatures:
    @staticmethod
    def compute_divergence_features(
        external_probs: dict,
        vision_probs: dict,
        ml_probs: dict | None = None,
    ) -> dict:
        ext_yes = float(external_probs.get("yes", 0.5))
        ext_no = float(external_probs.get("no", 0.5))
        vis_yes = float(vision_probs.get("yes", 0.5))
        vis_no = float(vision_probs.get("no", 0.5))

        out: dict = {
            "ext_prob_yes": ext_yes,
            "ext_prob_no": ext_no,
            "vision_prob_yes": vis_yes,
            "vision_prob_no": vis_no,
        }

        kl = 0.0
        for p, q in ((ext_yes, vis_yes), (ext_no, vis_no)):
            p_ = p + EPS
            q_ = q + EPS
            kl += p_ * math.log(p_ / q_)
        out["kl_div_ext_vision"] = kl

        div_yes = ext_yes - vis_yes
        div_no = ext_no - vis_no
        out["divergence_yes"] = div_yes
        out["divergence_no"] = div_no
        out["abs_divergence_yes"] = abs(div_yes)
        out["abs_divergence_no"] = abs(div_no)
        out["max_divergence"] = max(abs(div_yes), abs(div_no))

        ext_arg = "yes" if ext_yes >= ext_no else "no"
        vis_arg = "yes" if vis_yes >= vis_no else "no"
        out["sources_agree"] = 1 if ext_arg == vis_arg else 0

        out["blended_prob_yes"] = 0.5 * ext_yes + 0.5 * vis_yes
        out["blended_prob_no"] = 0.5 * ext_no + 0.5 * vis_no

        if ml_probs is not None:
            ml_yes = float(ml_probs.get("yes", 0.5))
            ml_no = float(ml_probs.get("no", 0.5))
            out["ml_prob_yes"] = ml_yes
            out["ml_prob_no"] = ml_no
            out["ml_vs_ext_yes"] = ml_yes - ext_yes
            out["ml_vs_ext_no"] = ml_no - ext_no
            out["ml_vs_vision_yes"] = ml_yes - vis_yes
            out["ml_vs_vision_no"] = ml_no - vis_no
            out["triple_blend_yes"] = 0.40 * ml_yes + 0.35 * vis_yes + 0.25 * ext_yes
            out["triple_blend_no"] = 0.40 * ml_no + 0.35 * vis_no + 0.25 * ext_no
            ml_arg = "yes" if ml_yes >= ml_no else "no"
            out["all_three_agree"] = 1 if (ext_arg == vis_arg == ml_arg) else 0

        return out


__all__ = ["TripleLayerFeatures"]
