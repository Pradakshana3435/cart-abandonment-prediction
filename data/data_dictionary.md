# Data Dictionary — Online Shoppers Purchasing Intention Dataset

| Column | Type | Description |
|---|---|---|
| Administrative | int | Number of admin/account pages visited |
| Administrative_Duration | float | Seconds spent on admin pages |
| Informational | int | Number of informational pages visited |
| Informational_Duration | float | Seconds spent on informational pages |
| ProductRelated | int | Number of product pages visited |
| ProductRelated_Duration | float | Seconds spent on product pages |
| BounceRates | float | Avg. bounce rate of pages visited (0-0.2) |
| ExitRates | float | Avg. exit rate of pages visited (0-0.2) |
| PageValues | float | Avg. value of pages visited before a transaction (analytics-derived; strongest predictor in this model) |
| SpecialDay | float | Closeness of visit to a special day (0 = far, 1 = very close) |
| Month | categorical | Month of visit |
| OperatingSystems | int (code) | OS identifier, not human-readable |
| Browser | int (code) | Browser identifier, not human-readable |
| Region | int (code) | Region identifier, not human-readable |
| TrafficType | int (code) | Traffic source identifier, not human-readable |
| VisitorType | categorical | New_Visitor / Returning_Visitor / Other |
| Weekend | bool | Whether session occurred on a weekend |
| Revenue | bool | **Target.** True = purchase completed |

## Deviation from original spec
This dataset contains no cart-specific fields (cart value, items added,
checkout-page-visited, shipping cost). `Revenue = False` was used as the
proxy for "abandoned." Two spec fields were approximated:
- **High cart value** → proxied by `ProductRelated` ≥ 75th percentile
- **Checkout visited** → proxied by `PageValues > 0`
No real proxy exists for cart value in currency or shipping cost;
these fields are absent from the model entirely.

## Abandonment window
Not time-windowed — `Revenue` reflects the single session's outcome only,