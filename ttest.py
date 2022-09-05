

match_box = [[None for i in range(5)] for j in range(20)] 

print(match_box)

ass = [2,3,4,5,6,7,9]

for idx, a in enumerate(ass):
    if a > 2:
        ass.pop(idx)

    print(ass)


ass = [2,3,4,5,6,7,9]

result = [v for v in ass if v > 2]

print(result)