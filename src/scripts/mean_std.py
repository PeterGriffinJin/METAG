import numpy as np
from decimal import Decimal

nums = [[[0.4935,0.4778,0.4778],	[0.4112,0.4095,0.4095],	[0.5343,0.5385,0.5385],	[0.6243,0.6347,0.6347],	[0.7151,0.7353,0.7353]],
[[0.5050,0.5079,0.5009],	[0.4360,0.4418,0.4348],	[0.5610,0.5729,0.5603],	[0.6557,0.6716,0.6524],	[0.7487,0.7652,0.7476]],
[[0.5091,0.5143,0.5071],	[0.4354,0.4438,0.4408],	[0.5604,0.5747,0.5689],	[0.6599,0.6780,0.6659],	[0.7551,0.7762,0.7641]],
[[0.5146,0.5182,0.5156],	[0.4469,0.4515,0.4513],	[0.5741,0.5814,0.5781],	[0.6733,0.6822,0.6748],	[0.7684,0.7784,0.7696]],
[[0.5181,0.5201,0.5154],	[0.4521,0.4557,0.4507],	[0.5779,0.5857,0.5763],	[0.6764,0.6876,0.6749],	[0.7716,0.7847,0.7732]],
[[0.5166,0.5182,0.5157],	[0.4512,0.4547,0.4536],	[0.5771,0.5838,0.5783],	[0.6761,0.6854,0.6777],	[0.7708,0.7824,0.7718]],
[[0.5151,0.5182,0.5142],	[0.4520,0.4531,0.4522],	[0.5823,0.5847,0.5805],	[0.6829,0.6890,0.6802],	[0.7780,0.7875,0.7759]],
[[0.5188,0.5197,0.5174],	[0.4563,0.4536,0.4551],	[0.5864,0.5848,0.5838],	[0.6889,0.6889,0.6838],	[0.7838,0.7883,0.7786]],
[[0.5185,0.5164,0.5171],	[0.4583,0.4543,0.4567],	[0.5859,0.5842,0.5829],	[0.6880,0.6898,0.6818],	[0.7832,0.7901,0.7784]],
[[0.5199,0.5187,0.5159],	[0.4620,0.4580,0.4568],	[0.5910,0.5900,0.5847],	[0.6941,0.6953,0.6838],	[0.7883,0.7941,0.7802]],
[[0.5211,0.5198,0.5166],	[0.4638,0.4582,0.4603],	[0.5914,0.5892,0.5874],	[0.6919,0.6930,0.6844],	[0.7859,0.7911,0.7792]],
[[0.5214,0.5188,0.5173],	[0.4650,0.4577,0.4601],	[0.5937,0.5884,0.5856],	[0.6959,0.6940,0.6822],	[0.7896,0.7922,0.7777]],
[[0.5207,0.5192,0.5176],	[0.4658,0.4591,0.4622],	[0.5944,0.5899,0.5879],	[0.6965,0.6947,0.6841],	[0.7902,0.7927,0.7794]]]


for ns in nums:
    print(f'{Decimal(np.mean(ns[0])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")}±{Decimal(np.std(ns[0])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")} {Decimal(np.mean(ns[1])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")}±{Decimal(np.std(ns[1])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")} {Decimal(np.mean(ns[2])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")}±{Decimal(np.std(ns[2])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")} {Decimal(np.mean(ns[3])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")}±{Decimal(np.std(ns[3])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")} {Decimal(np.mean(ns[4])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")}±{Decimal(np.std(ns[4])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")}')
    # print(f'{Decimal(np.mean(ns[0])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")}±{Decimal(np.std(ns[0])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")} {Decimal(np.mean(ns[1])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")}±{Decimal(np.std(ns[1])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")} {Decimal(np.mean(ns[2])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")}±{Decimal(np.std(ns[2])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")} {Decimal(np.mean(ns[3])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")}±{Decimal(np.std(ns[3])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")} {Decimal(np.mean(ns[4])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")}±{Decimal(np.std(ns[4])).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")}')
