import numpy as np

domains_acc = [0.8935, 0.9441, 0.9128, 0.8791, 0.7140, 0.5145]
individual_acc = []
decay_speed = 1.6
domain_num = len(domains_acc)
client_nums = [0 for _ in range(domain_num)]
for i in range(domain_num):
    client_nums[i] = round(np.float_power(decay_speed, domain_num-i-1))

print(client_nums)
for i, acc in enumerate(domains_acc):
    for val in range(client_nums[i]):
        individual_acc.append(acc)

std_domain = np.std(domains_acc, dtype=np.float64)
std_individual = np.std(individual_acc, dtype=np.float64)
mean_domain = np.mean(domains_acc)
mean_individual = np.mean(individual_acc)
msg = \
        f"mean_domain: {mean_domain:.4f}, " \
        + f"mean_individual: {mean_individual:.4f},\n" \
        + f"std_domain: {std_domain:.4f}, " \
        + f"std_individual: {std_individual:.4f}, " 

print(msg)