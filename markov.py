import random


# --------------------------------------------------------------
# 1. МЕТОД 1: ГРУППИРОВКА С РАВНЫМИ ИНТЕРВАЛАМИ
# --------------------------------------------------------------
def group_equal_intervals(X, k=2):
    """
    Группировка с равными интервалами.
    Возвращает список из k списков значений.
    """
    Xmin = min(X)
    Xmax = max(X)
    shag = (Xmax - Xmin) / k

    groups = [[] for _ in range(k)]
    for x in X:
        # Определяем номер группы
        if x == Xmax:
            idx = k - 1                     # последнее значение — в последнюю группу
        else:
            idx = int((x - Xmin) / shag)
        groups[idx].append(x)
    return groups


# --------------------------------------------------------------
# 2. МЕТОД 2: ГРУППИРОВКА С РАВНЫМ ЧИСЛОМ ЭЛЕМЕНТОВ
# --------------------------------------------------------------
def group_equal_size(X, k=2):
    """
    Группировка с равным числом элементов (по квантилям).
    Возвращает список из k списков значений.
    """
    sorted_X = sorted(X)
    n = len(sorted_X)
    base_size = n // k
    remainder = n % k

    groups = []
    start = 0
    for i in range(k):
        # Первые `remainder` групп получают на 1 элемент больше
        size = base_size + (1 if i < remainder else 0)
        groups.append(sorted_X[start:start + size])
        start += size
    return groups


# --------------------------------------------------------------
# 3. ФУНКЦИЯ СРАВНЕНИЯ
# --------------------------------------------------------------
def compare_grouping(name, X, k=2):
    print("=" * 72)
    print(f"НАБОР {name}: {len(X)} значений, число групп k = {k}")
    print("=" * 72)

    g1 = group_equal_intervals(X, k)
    g2 = group_equal_size(X, k)

    def describe(label, groups, X):
        Xmin = min(X)
        Xmax = max(X)
        shag = (Xmax - Xmin) / k

        print(f"\n{label}")
        print("-" * 72)
        print(f"{'Группа':<8} | {'Размер':>8} | {'Мин':>8} | {'Макс':>8} | {'Среднее':>10} | {'Границы':<20}")
        print("-" * 72)

        for i, g in enumerate(groups, 1):
            if not g:
                continue
            if label.startswith("Равные"):
                lo = Xmin + (i - 1) * shag
                hi = Xmin + i * shag
                bounds = f"[{lo:.2f}; {hi:.2f}]"
            else:
                bounds = f"[{min(g)}; {max(g)}]"
            print(f"{i:<8} | {len(g):>8} | {min(g):>8} | {max(g):>8} | {sum(g)/len(g):>10.2f} | {bounds:<20}")

    describe("Метод 1: Равные интервалы", g1, X)
    describe("Метод 2: Равное число элементов (квартили)", g2, X)

    # Равномерность распределения
    sizes1 = [len(g) for g in g1]
    sizes2 = [len(g) for g in g2]
    spread1 = max(sizes1) - min(sizes1)
    spread2 = max(sizes2) - min(sizes2)

    print()
    print(f"Разброс размеров групп (равные интервалы):  {spread1}")
    print(f"Разброс размеров групп (равное число):      {spread2}")
    print(f"Метод с равным числом даёт более равномерное распределение: {spread2 <= spread1}")
    print()


# --------------------------------------------------------------
# 4. ТЕСТОВЫЕ НАБОРЫ
# --------------------------------------------------------------
dataset_A = [22, 25, 23, 22, 27, 22, 26]

random.seed(42)
dataset_B = [random.uniform(15, 35) for _ in range(1000)]


# --------------------------------------------------------------
# 5. ЗАПУСК
# --------------------------------------------------------------
compare_grouping("А (7 доходов из задания 2.4)", dataset_A, k=2)
compare_grouping("Б (1000 случайных доходов)", dataset_B, k=2)