label = ["3", "3 Dup", "3", "4", "5", "5 Dup", "5"]
result = [10, 5 , 20, 22 , 2, 3, 10]

# Create dict
dict = {'Depth': label,
			'PCE': result}

# Create DataFrame from dict
import pandas as pd
df = pd.DataFrame(dict)
print(df)

depth = list(df.Depth)
pce = list(df.PCE)

print(depth, pce)

Duplicate = []
PCE2 = []
for i in range(len(depth)):
	if "Dup" in depth[i]:
		Duplicate.append("Yes")
		PCE2.append(pce[i-1])
		print(i)
	else:
		Duplicate.append("No")
		PCE2.append(pce[i])
		print(i)
print(Duplicate, PCE2)