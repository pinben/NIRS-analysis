import math

def adjust_d_model(d_model, channel_nhead, nhead):
    lcm = abs(channel_nhead * nhead) // math.gcd(channel_nhead, nhead)
    return (d_model // lcm) * lcm