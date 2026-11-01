# 4.2.4.4 Prediction Explanation and Wafer Grading

Once the prediction output is obtained, the system produces a model explanation and calculates the local wafer grade. SHAP is applied to identify the input features that contribute to the model prediction. Meanwhile, the grade is calculated from the similarity scores and predicted yield of the local electrical-test workflow. The SHAP values are interpreted separately from the grade calculation, and the local grade is also kept separate from the Bosch and equipment benchmark scores.

**SHAP-based prediction explanation**

For each prediction, the SHAP value represents the contribution of an input feature relative to the model's reference output. A positive SHAP value indicates that the feature increases the prediction, whereas a negative value indicates that it decreases the prediction. The feature name, prediction target and fitted model are displayed to support interpretation. The highlighted features can then be compared with the measured parameters to guide engineering investigation. However, the contributions do not directly confirm the physical cause of a defect.

For Bosch, the beeswarm plot is generated using a LightGBM model fitted to the full matched dataset. Although LightGBM is also the saved wafer-average silicon-etch champion, the plot is produced from a separate model refitted on the complete data. Meanwhile, the reported R² is calculated from predictions of the excluded validation lots. Therefore, the beeswarm is used to explain the fitted model, while the grouped validation results are used to assess its prediction performance.

**Similarity score and yield penalty**

Following prediction, the Wafer Probe and predicted Final Test similarities are combined with equal weights to calculate the base score. If the predicted yield is below 0.80, the difference from 0.80 is multiplied by 100 to obtain the yield penalty. The penalty is limited to 25 points and becomes zero when the predicted yield reaches the threshold. This penalty is then subtracted from the base score, and the final value is limited to the range from zero to 100.

Base score = 0.5 × probe similarity + 0.5 × final-test similarity

Final score = clip(base score - yield penalty, 0, 100)

For example, a Wafer Probe similarity of 92 and a predicted Final Test similarity of 84 produce an average score of 88. If the predicted yield is 0.75, a penalty of five points is applied. Therefore, the final score becomes 83 and is initially assigned to Grade B before the additional grading rules are applied.

**Grade boundaries and review of the result**

| Grade | Initial score band | Displayed label |
|---|---|---|
| A | 90 to 100 | Automotive |
| B | 80 to below 90 | Industrial |
| C | 60 to below 80 | Consumer |
| D | Below 60 | Scrap |

The implemented Grade B boundary follows Phase 1 Table 3-6 at 80. Predicted yield below 0.60 forces the result to Grade D. When a numeric golden correlation is available, a value below 0.20 also forces Grade D.

However, the mean time to failure (MTTF) input is currently accepted without being included in the score or grade calculation. Besides, the Z-score-closeness function returns 100 when an eligible reference comparison is unavailable. Therefore, the reference coverage should be checked before the grade is interpreted. The Automotive, Industrial, Consumer and Scrap labels assigned to Grades A, B, C and D are used as preliminary screening categories rather than formal product qualifications.

![Figure 4.9: Review sequence for local prediction explanations and advisory grades](../output/pdf/section_4.2.4.4/figure_4.9_explanation_and_grading.png)

Figure 4.9: Proposed engineer review sequence for local explanations and advisory grades.

SHAP and grading are separate calculations. The final review boxes describe the proposed human review process, not an automatic material-release decision.

# 4.2.4.5 AI-Assisted Report and Shift Handover

Once the local analysis is completed, the calculated results are passed to the AI report-handover function to prepare a shift note. The note summarises the wafer condition so that the next shift can review the prediction and the supporting measurements. This function is applied after model prediction. The AutoML planner selects the search profile, the regression model calculates the prediction, and the report function converts the supplied results into a written explanation without changing the prediction or wafer grade.

**Preparing the handover information**

Firstly, the function receives the wafer identity, product family, part number, predicted yield, historical average yield, reliability grade, defect pattern, failed-die count, total-die count and selected electrical outliers. These results are used to provide the wafer and product information required for the handover note. The function prepares the report from the supplied analysis rather than performing another wafer measurement or independently identifying the defect pattern.

Before the prompt is prepared, fractional yield values are converted into percentages. The difference between the predicted and historical yields and the percentage of failed dies are then calculated. For example, predicted yield of 0.85 and historical yield of 0.90 are converted to 85 percent and 90 percent. The resulting difference is minus five percentage points. These values are inserted into the prompt together with the wafer and product information.

**Groq request and retry settings**

Next, the prompt requests a shift summary, a possible process explanation and suggested follow-up actions. When AI generation is enabled and credentials are available, the prompt is submitted to the Groq chat-completions endpoint. The request uses temperature 0.25, a maximum of 350 output tokens and a timeout of 12 seconds per request. These values are the request settings used by the function and do not represent measured response times.

The function attempts an ordered list of three model identifiers. If a request fails, the next model identifier is selected and the request is repeated. Once a successful HTTP response is received and parsed, the function returns the generated note with its model source and success status. The availability of the configured models has not been verified through a live-service test during this report revision.

Following report generation, the note and its source are displayed in the interface and can be downloaded as a text file. The engineer can then compare the explanation with the numerical results and source measurements before deciding on the follow-up investigation. The handover function does not send commands to production equipment. Its generation checks and remaining limitations are discussed in Chapter 5.

**Fallback output and engineer review**

If AI generation is disabled, credentials are unavailable or all candidate requests fail, the function produces a deterministic template and returns a fallback status. The supplied results are combined with predefined wording for edge, ring, scratch and other patterns to prepare the note. The source and status are displayed so that the user can identify whether the report was generated by Groq or by the template. Figure 4.10 illustrates the request, retry and fallback sequence.

![Figure 4.10: AI-assisted shift handover, model retry and template fallback](../output/pdf/section_4.2.4.5/figure_4.10_ai_handover.png)

Figure 4.10: AI-assisted handover generation, candidate-model retry and template fallback.

However, the current prompt describes predicted yield as measured yield, while the template associates defect patterns with causes and equipment actions that have not been established from the supplied data. Therefore, both forms of the handover note require engineering review. Predicted values should be identified as predictions, and the suggested causes should be verified through further investigation. Confidential production information also requires approval before it is submitted to the external service.
