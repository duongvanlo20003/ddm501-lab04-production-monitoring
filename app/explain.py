"""
SHAP explanations for individual scoring decisions.

"""

import logging
from typing import Any, Dict, List

import numpy as np

from pipeline.preprocessing import add_derived_features

logger = logging.getLogger(__name__)


class Explainer:
    """Wraps a SHAP TreeExplainer around the fitted pipeline.

    The pipeline is (features -> classifier). SHAP needs the raw estimator and
    the TRANSFORMED matrix, so this class holds both halves and does the
    transformation itself. Handing the whole pipeline to shap and hoping is
    the usual mistake.
    """

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.feature_step = pipeline.named_steps["features"]
        self.classifier = pipeline.named_steps["classifier"]
        self.feature_names = list(
            self.feature_step.named_steps["preprocess"].get_feature_names_out()
        )
        self._explainer = None

    def _ensure_explainer(self):
        """Build the explainer on first use.

        Lazily, because importing shap costs about a second and a container
        that is not asked for explanations should not pay it at startup.
        """
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.classifier)
        return self._explainer

    def explain(self, frame, top_n: int = 8) -> Dict[str, Any]:
        """Return the features that moved this one score, largest first.

        TASK 10:
          - Transform the frame with self.feature_step, then call
            shap_values on self.classifier's explainer.
          - Depending on the model, shap returns (n, features) or
            (n, features, classes). Normalise to the positive class.
          - expected_value may be a scalar or an array; normalise it too.
          - Sort by ABSOLUTE contribution and keep the top n.
          - Report the applicant's OWN value for each feature, not the
            standardised one. add_derived_features(frame) gives you the
            derived ones; one-hot columns have no counterpart in the
            application and fall back to the transformed value. An adverse
            action notice quoting "your PAY_0 was 2.16" cannot be reconciled
            with the application form by anyone outside the ML team.
        """
        transformed = np.asarray(self.feature_step.transform(frame))
        explainer = self._ensure_explainer()
        raw_values = explainer.shap_values(transformed)
        if isinstance(raw_values, list):
            raw_values = raw_values[-1]
        values = np.asarray(raw_values)
        if values.ndim == 3:
            contributions = values[0, :, -1]
        elif values.ndim == 2:
            contributions = values[0]
        else:
            contributions = values.reshape(-1)

        expected = np.asarray(explainer.expected_value)
        base_value = float(expected.reshape(-1)[-1])
        own_values = add_derived_features(frame).iloc[0]

        items: List[Dict[str, Any]] = []
        for index, (feature, contribution) in enumerate(
            zip(self.feature_names, contributions)
        ):
            if feature in own_values.index:
                value = own_values[feature]
            else:
                value = transformed[0, index]
            items.append({
                "feature": str(feature),
                "value": float(value),
                "contribution": float(contribution),
                "direction": "increases risk" if contribution >= 0 else "reduces risk",
            })
        items.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return {"base_value": base_value, "contributions": items[:top_n]}
