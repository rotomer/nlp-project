
class Pipeline(object):
    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass


# Flow:
# 1. train start with parsing and loading all of the eps data, and all of the prices data and index it per share.
# 2. build predictor to predict up or down.
# 3. calculate precision metrics for 1 time range. Allow different time ranges in the future:
#     (1,2,3,4,5 days, 2 week, 1 month, 3 months, 1 year, 3 years)

# -- Target date 24-8 EoD

# Regular Sentiment Analysis:
# Start with one stock - say AAPL, then all of them:
# 1. Get sentiment per form.
# 2. Predict using sentiment.
# 3. Calculate precision change for time range(s).

# -- Target date 25-8 EoD

# Relation extraction to 8k forms:
# Start with one stock - say AAPL, then all of them:
# 1. Add features from the extracted rules.
# 2. Add to prediction.
# 3. Calculate precision change for time range(s).

# -- Target date 27-8 EoD

# Relation extraction to 10k forms:
# Start with one stock - say AAPL, then all of them:
# 1. Fetch data from edgar.
# 2. Add features from the extracted rules.
# 3. Add to prediction.
# 4. Calculate precision change for time range(s).

# -- Target date 30-8 EoD
