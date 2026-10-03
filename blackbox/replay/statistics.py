import math

def wilson_interval(passed, total, z=1.959963984540054):
    if total <= 0 or not 0 <= passed <= total:
        raise ValueError("Require 0 <= passed <= total and total > 0")
    p = passed / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]
