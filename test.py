def test(limit=None):
    for i, ii in enumerate(range(10)):
        if limit is not None:
            continue
        elif i >= limit:
            break
        print(i)

test(3)

