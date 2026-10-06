# Refund Reconciliation and GW-OTHER Review

**To:** Arjun Mehta, Finance Controller  
**Subject:** Refund reconciliation, GW-OTHER exposure, and review priorities

## Executive Summary

The support-ticket export materially overstated refund values because legacy Freshdesk monetary values were stored in a different unit and some tickets appeared more than once following migration/re-import. After normalizing legacy values and retaining the current helpdesk record where the same ticket appeared in both systems, the dataset contains **11,600 unique tickets and ₹67.10 lakh of refunds across 2,340 refund tickets**.

The cleaned monthly, reason-level, and agent-level refund views reconcile to the same **₹67,09,932** total.

## Where the refund value is concentrated

`GW-OTHER` is the largest refund reason by value:

- **991 tickets**
- **₹29.07 lakh**
- **43.32% of total refund value**

This makes `GW-OTHER` the clearest area for improving refund-reason accuracy and operational visibility.

The investigation shows that these refunds are distributed across Returns, Billing, Frontline, Logistics and other teams rather than being isolated to one group.

## AI-assisted classification opportunity

A text-classification model was evaluated on known refund reasons and achieved:

- **92.96% accuracy**
- **86.92% macro F1**

When applied to `GW-OTHER`, **220 tickets representing ₹6.27 lakh of refund value** meet the selected ≥80% confidence threshold.

This should be treated as **classifiable refund value, not guaranteed savings**. High-confidence cases can be used to suggest a more specific refund reason, while lower-confidence cases should continue to manual review.

The current manual-review queue contains **771 lower-confidence cases**.

## Policy review

The order-level audit identified **192 orders** with both refund and replacement activity, involving **₹6.50 lakh of refund value**.

Of these:

- **100** involve refund and replacement activity on the same ticket.
- **92** involve the activity across multiple tickets for the same order.

These are investigation flags rather than automatic conclusions of policy violation. Operations should review the underlying ticket history before taking corrective action.

## Recommended actions

1. **Use the cleaned refund total as the finance reporting baseline** rather than the raw export total.
2. **Review the 192 refund-plus-replacement orders**, prioritizing the highest-value cases.
3. **Pilot AI-assisted coding for the 220 high-confidence GW-OTHER tickets** while keeping final approval with support/finance staff.
4. **Keep lower-confidence cases in manual review** and track model accuracy after deployment.
5. **Monitor GW-OTHER monthly** because it currently represents 43.32% of refund value.
6. **Measure actual financial impact after intervention** before claiming any savings.

## Bottom line

The immediate finance issue is not simply that refunds increased. The export contains reconciliation problems that can materially distort the reported value. After correction, the refund pool is **₹67.10 lakh**, with **₹29.07 lakh concentrated in GW-OTHER**.

The proposed workflow gives Finance and Support a controlled way to improve classification accuracy, focus manual investigation on the highest-risk cases, and establish a measurable baseline for future refund leakage or processing-cost improvements.