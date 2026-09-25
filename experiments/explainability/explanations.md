# Day 10 Prediction Explanations

SHAP was attempted but could not be installed because this Python environment requires Microsoft C++ Build Tools to build SHAP. Contributions are computed exactly as TF-IDF value times the saved Logistic Regression coefficient for the predicted class.

Predictions explained: 5
Confidence range: 0.174072 to 0.996855

## KAGGLE-f187c090e595-008850

- Actual category: Returns and Exchanges
- Predicted category: Billing and Payments
- Confidence: 0.174072
- Second-best: Customer Service (0.169047)
- Summary: Predicted Billing and Payments with model probability 0.174. Strongest supporting terms: action, security, action to. Strongest opposing terms: training, staff, infrastructure.

Top supporting features:
- action: 0.254119
- security: 0.252036
- action to: 0.208206
- safeguard medical: 0.190586
- this concern: 0.186070
- is critical: 0.183496
- immediate action: 0.179801
- including: 0.156644

Top opposing features:
- training: -0.212907
- staff: -0.165195
- infrastructure: -0.160290
- improve: -0.141895
- steps: -0.120610
- hospital infrastructure: -0.119721
- sensitive patient: -0.118975
- to improve: -0.110158

## KAGGLE-f187c090e595-026513

- Actual category: IT Support
- Predicted category: Technical Support
- Confidence: 0.447547
- Second-best: Product Support (0.363595)
- Summary: Predicted Technical Support with model probability 0.448. Strongest supporting terms: integration issues, their investment, issues. Strongest opposing terms: financial, financial company, issues across.

Top supporting features:
- integration issues: 0.321227
- their investment: 0.307632
- issues: 0.243530
- clearing caches: 0.213102
- platforms which: 0.212877
- rebooting devices: 0.212549
- address this: 0.199594
- processes recent: 0.187484

Top opposing features:
- financial: -0.295598
- financial company: -0.228172
- issues across: -0.208561
- have impacted: -0.174045
- issues in: -0.171789
- and clearing: -0.157451
- their: -0.147912
- impeding: -0.143297

## KAGGLE-f187c090e595-020230

- Actual category: IT Support
- Predicted category: IT Support
- Confidence: 0.593814
- Second-best: Customer Service (0.128109)
- Summary: Predicted IT Support with model probability 0.594. Strongest supporting terms: need guidance, protocols to, firebase. Strongest opposing terms: data, data for, capabilities to.

Top supporting features:
- need guidance: 0.499142
- protocols to: 0.391446
- firebase: 0.274046
- support assistance: 0.266464
- information looking: 0.233545
- to streamline: 0.183840
- confidentiality and: 0.166998
- and integrity: 0.164906

Top opposing features:
- data: -0.402846
- data for: -0.202800
- capabilities to: -0.193854
- in securing: -0.183582
- protocols: -0.163242
- securing medical: -0.160455
- medical data: -0.151276
- medical: -0.103197

## KAGGLE-f187c090e595-019967

- Actual category: Product Support
- Predicted category: Product Support
- Confidence: 0.773817
- Second-best: Technical Support (0.185598)
- Summary: Predicted Product Support with model probability 0.774. Strongest supporting terms: investment analytics, system configurations, slow performance. Strongest opposing terms: server resources, its investment, tool the.

Top supporting features:
- investment analytics: 0.357831
- system configurations: 0.314520
- slow performance: 0.313511
- large datasets: 0.282594
- to improper: 0.195511
- or resource: 0.193271
- processing large: 0.191570
- slow: 0.189278

Top opposing features:
- server resources: -0.177421
- its investment: -0.161738
- tool the: -0.142483
- resources: -0.130459
- but: -0.119443
- configurations or: -0.117075
- system: -0.116454
- still persist: -0.115107

## KAGGLE-f187c090e595-022784

- Actual category: Service Outages and Maintenance
- Predicted category: Service Outages and Maintenance
- Confidence: 0.996855
- Second-best: Technical Support (0.000912)
- Summary: Predicted Service Outages and Maintenance with model probability 0.997. Strongest supporting terms: service, outages, financial. Strongest opposing terms: investment, with, data.

Top supporting features:
- service: 2.379536
- outages: 1.782789
- financial: 0.656216
- downtime: 0.531244
- service outages: 0.493593
- disruptions: 0.417689
- impacted: 0.352535
- outages that: 0.306746

Top opposing features:
- investment: -0.200092
- with: -0.132969
- data: -0.095425
- problem: -0.092054
- that impacted: -0.081849
- financial company: -0.064644
- the: -0.047446
- persist: -0.045672
