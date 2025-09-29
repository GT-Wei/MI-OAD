categories = {
    "aircraft":15792,
    "aircraft-hangar":221,
    "airplane":58608,
    "airport":2884,
    "awning-tricycle":7893,
    "barge":208,
    "baseball-diamond":9051,
    "baseballfield":5818,
    "basketball-court":11323,
    "bicycle":21960,
    "bridge":13886,
    "building":381599,
    "bus":21220,
    "car":302557,
    "cargo-car":2266,
    "cargo-plane":786,
    "cargo-truck":7359,
    "cement-mixer":392,
    "chimney":1680,
    "construction-site":1190,
    "container":157558,
    "container-crane":507,
    "container-ship":322,
    "crane-truck":209,
    "crossroad":8974,
    "dam":1050,
    "damaged-building":1365,
    "dump-truck":1692,
    "engineering-vehicle":257,
    "excavator":1032,
    "expressway-service-area":2165,
    "expressway-toll-station":1298,
    "facility":991,
    "ferry":236,
    "fishing-vessel":945,
    "fixed-wing-aircraft":84,
    "flat-car":164,
    "front-loader/bulldozer":775,
    "golffield":1086,
    "ground-grader":96,
    "ground-track-field":10764,
    "harbor":25563,
    "haul-truck":384,
    "helicopter":2701,
    "helipad":269,
    "hut/tent":921,
    "large-vehicle":69855,
    "locomotive":136,
    "maritime-vessel":808,
    "mobile-crane":386,
    "motor":70227,
    "motorboat":1823,
    "oil-tanker":5224,
    "overpass":3806,
    "parking-lot":8710,
    "passenger-car":1869,
    "passenger-vehicle":3831,
    "pedestrian":173321,
    "people":65996,
    "pickup-truck":1379,
    "plane":14217,
    "playground":590,
    "pylon":437,
    "railway-vehicle":21,
    "reach-stacker":79,
    "roundabout":1329,
    "sailboat":816,
    "scraper/tractor":96,
    "shed":1530,
    "ship":225869,
    "shipping-container":1846,
    "shipping-container-lot":2465,
    "small-aircraft":448,
    "small-car":263128,
    "small-vehicle":906075,
    "soccer-ball-field":709,
    "stadium":1267,
    "storage-tank":67684,
    "storagetank":26403,
    "straddle-carrier":72,
    "swimming-pool":37569,
    "t-junction":8716,
    "tank-car":143,
    "tennis-court":15375,
    "tenniscourt":12241,
    "tower":106,
    "tower-crane":172,
    "trailer":5051,
    "trainstation":1010,
    "tricycle":10864,
    "truck":40969,
    "truck-tractor":1032,
    "truck-w/box":4248,
    "truck-w/flatbed":1120,
    "truck-w/liquid":183,
    "tugboat":261,
    "utility-truck":4470,
    "van":49169,
    "vehicle":49137,
    "vehicle-lot":4729,
    "windmill":37718,
    "yacht":547
}

# 1. 将类别按频次升序排序
sorted_categories = sorted(categories.items(), key=lambda x: x[1])

# 将排序结果保存到1.txt中，类别前加编号
with open("1.txt", "w") as f:
    for i, (cat, freq) in enumerate(sorted_categories, start=1):
        f.write(f"{i}. {cat}: {freq}\n")

# 2. 确定分位点（共102类，分成3等份：rare前34个，common中间34个，frequent最后34个）
total_count = len(sorted_categories)
one_third = total_count // 3
rare_cutoff = one_third
common_cutoff = one_third * 2

rare_categories = sorted_categories[:rare_cutoff]
common_categories = sorted_categories[rare_cutoff:common_cutoff]
frequent_categories = sorted_categories[common_cutoff:]

# 3. 将三个类别分别根据实例数从大到小排序，并写入文件，同时加上频次
rare_categories_desc = sorted(rare_categories, key=lambda x: x[1], reverse=True)
common_categories_desc = sorted(common_categories, key=lambda x: x[1], reverse=True)
frequent_categories_desc = sorted(frequent_categories, key=lambda x: x[1], reverse=True)

with open("rare.txt", "w") as f:
    for i, (cat, freq) in enumerate(rare_categories_desc, start=1):
        f.write(f"{i}. {cat}: {freq}\n")

with open("common.txt", "w") as f:
    for i, (cat, freq) in enumerate(common_categories_desc, start=1):
        f.write(f"{i}. {cat}: {freq}\n")

with open("frequent.txt", "w") as f:
    for i, (cat, freq) in enumerate(frequent_categories_desc, start=1):
        f.write(f"{i}. {cat}: {freq}\n")
