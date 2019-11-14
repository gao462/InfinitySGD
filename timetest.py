import ddmp
import torch

# generate tensors
N   = 200
K   = 20
MIN = 1
MAX = 51

# forward and backward
for _ in range(N):
    input = torch.nn.Parameter(torch.LongTensor(1, K, K).random_(MIN, MAX).double())
    for i in range(K):
        input.data[:, i, i] = 0
    _, output, _ = ddmp.stdy_dist(input)
    loss = torch.norm(output)
    loss.backward()

# output table
MAXLEN = max([len(itr) for itr in ddmp.TIME_COST.keys()])
ALL = sum([sum(itr) for itr in ddmp.TIME_COST.values()])
HEAD = "{name:<{len}s}    {avg:>13s}    {cnt:>5s}    {total:>10s}".format(
    name='', len=MAXLEN, avg='Average (sec)', cnt='#Hit', total='Total (sec)')
print('Time Cost Statistics')
print()
print("Total Time: {:.3f} sec".format(ALL))
print()
print(HEAD)
print("=" * len(HEAD))
for key in ddmp.TIME_COST:
    data = ddmp.TIME_COST[key]
    total = sum(data)
    assert len(data) == N
    average = total / N
    print("{name:<{len}s}    {avg:>13s}    {cnt:>5d}    {total:.3f}".format(
        name=key, len=MAXLEN, avg="{:.7f}".format(average), cnt=N, total=total))
