import collections


def get_dist(pos1, pos2):
    return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2


def get_new_center(pos_list):
    center_x, center_y = 0, 0
    for pos in pos_list:
        center_x += pos[0]
        center_y += pos[1]
    return (center_x / len(pos_list), center_y / len(pos_list))


def k_means(position, k):
    class_center = position[:k]
    sample_class = [None for i in range(len(position))]
    iteration = 0

    while iteration < 100:
        class_cluster = collections.defaultdict(list)

        for i in range(len(position)):
            cur = position[i]
            min_distance = float('inf')
            tru_lable = -1

            for label in range(k):
                center = class_center[label]
                distance = get_dist(cur, center)
                if distance < min_distance:
                    min_distance = distance
                    tru_lable = label
            
            class_cluster[tru_lable].append(cur)
            sample_class[i] = tru_lable

        for label in class_cluster:
            new_center = get_new_center(class_cluster[label])
            class_center[label] = new_center

        iteration += 1

    return sample_class


if __name__ == '__main__':
    arr = [
        [1.5, 2.1],
        [0.8, 2.1],
        [1.3, 2.1],
        [110.5, 260.6],
        [21.7, 32.8],
        [130.9, 150.8],
        [32.6, 40.7],
        [41.5, 24.7]
    ]
    k = 3
    ans = k_means(arr, k)
    for cls in ans:
        print(cls)